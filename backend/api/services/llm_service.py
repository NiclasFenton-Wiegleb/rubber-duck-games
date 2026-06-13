"""
Small Language Model service.

Loads the locally hosted, fine-tuned SmolLM3-3B model
(`niclasfw/smollm3-3b-codex` by default) once and serves two kinds of output:

  - generate_simple():  a direct, fast answer (no extended reasoning). Implements
                        the full "simple request" workstream: file-based system
                        prompt + multi-KG context retrieval + SLM generation.
  - generate_complex(): a richer answer using SmolLM3's "thinking" mode and,
                        optionally, Knowledge-Graph / RAG context.

The model is heavy, so we keep a process-wide singleton and (by default) load
it lazily on the first request. Transformers/torch are imported lazily so the
Flask app can boot without them installed.
"""

from __future__ import annotations

import threading

# Process-wide singleton so the multi-GB model is only loaded once.
_INSTANCE: "LLMService | None" = None
_LOCK = threading.Lock()


def get_llm_service(config) -> "LLMService":
    """Return the shared LLMService, creating it on first use."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = LLMService(config)
                if not config["LLM_LAZY_LOAD"]:
                    _INSTANCE.load()
    return _INSTANCE


class LLMService:
    def __init__(self, config):
        self.config = config
        self.model_id = config["LLM_MODEL_ID"]
        self.device_pref = config["LLM_DEVICE"]
        self.default_max_new = config["LLM_MAX_NEW_TOKENS"]
        self.complex_max_new = config["LLM_COMPLEX_MAX_NEW_TOKENS"]
        self.default_temperature = config["LLM_TEMPERATURE"]

        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        # Resolved at load() time — the concrete device the model actually
        # ended up on (e.g. "cuda" pref may resolve to "cpu" with no GPU).
        self.device: str | None = None


    # ── Model loading ────────────────────────────────────────────────────────
    def load(self):
        """Load tokenizer + model into memory (idempotent)."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM

            device = self._resolve_device(torch)
            device_map = {"": device}
            dtype = self._dtype_for_device(torch, device)

            self._tokenizer = self._load_tokenizer()
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device_map,
            )
            # Record the concrete device the model actually landed on so callers
            # (e.g. the UI status chip) can report GPU vs. CPU accurately.
            self.device = device

    def _resolve_device(self, torch) -> str:
        """Pick the best available local device for model inference.

        An explicit preference is honoured *only if that accelerator is
        actually usable*. In particular a pinned ``"cuda"`` preference — which
        the ZeroGPU entry point sets unconditionally because a GPU is only
        attached inside a ``@spaces.GPU`` call — gracefully falls back to CPU
        when no GPU is present. This lets the same image run on both GPU and
        CPU-only instances without crashing on model load.
        """
        pref = (self.device_pref or "auto").lower()

        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if pref == "mps":
            mps = getattr(torch.backends, "mps", None)
            return "mps" if mps and mps.is_available() else "cpu"
        if pref == "cpu":
            return "cpu"

        # "auto" (or anything unrecognised): pick the best available device.
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


    @staticmethod
    def _dtype_for_device(torch, device: str):
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32

    def _load_tokenizer(self):
        """
        Load the tokenizer for ``self.model_id``.

        Some checkpoints (e.g. ones saved with a bleeding-edge / dev build of
        ``transformers``) write ``"tokenizer_class": "TokenizersBackend"`` into
        their ``tokenizer_config.json``. Older ``transformers`` releases don't
        know that class, so ``AutoTokenizer`` raises::

            ValueError: Tokenizer class TokenizersBackend does not exist or is
            not currently imported.

        We first try the normal ``AutoTokenizer`` path, then fall back to
        loading the fast tokenizer directly from ``tokenizer.json`` (which
        bypasses the ``tokenizer_class`` lookup entirely while still picking up
        the chat template and special tokens from ``tokenizer_config.json``).
        """
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(self.model_id)
        except (ValueError, KeyError):
            from transformers import PreTrainedTokenizerFast

            return PreTrainedTokenizerFast.from_pretrained(self.model_id)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Public generation methods ─────────────────────────────────────────────

    def generate_simple(self, query: str, max_new_tokens=None, temperature=None,
                         use_context=True) -> dict:
        """
        Direct, fast answer — no extended reasoning.

        Implements the full "simple request" workstream:
          1. system prompt is loaded from a local text file (cached),
          2. the most relevant context across all KGs in the storage bucket is
             retrieved and injected (skippable via use_context=False),
          3. the assembled prompt is run straight through the SLM,
          4. the answer + provenance is returned for the frontend.
        """
        from api.services.prompt_service import get_system_prompt

        system = get_system_prompt(self.config)

        context, sources = "", []
        if use_context:
            context, sources = self._retrieve_context(query)

        user = query
        if context:
            user = (
                "Use the following reference context if relevant:\n\n"
                f"{context}\n\nQuestion: {query}"
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        answer = self._chat(
            messages,
            max_new_tokens=max_new_tokens or self.default_max_new,
            temperature=temperature if temperature is not None else self.default_temperature,
        )
        return {
            "answer": answer.strip(),
            "reasoning": None,
            "context_used": bool(context),
            "sources": sources,
        }

    def generate_complex(self, query: str, use_context=True,
                         max_new_tokens=None, temperature=None) -> dict:
        """
        Richer answer: enables SmolLM3 reasoning ("/think") and optionally
        injects Knowledge-Graph / RAG context. Splits out the model's
        <think>...</think> trace into a separate `reasoning` field.
        """
        context, sources = ("", [])
        if use_context:
            context, sources = self._retrieve_context(query)

        system = (
            "You are an expert coding assistant. Think step by step, consider "
            "edge cases, and produce a thorough, well-structured answer with "
            "code examples where helpful. /think"
        )
        user = query
        if context:
            user = f"Use the following reference context if relevant:\n\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = self._chat(
            messages,
            max_new_tokens=max_new_tokens or self.complex_max_new,
            temperature=temperature if temperature is not None else self.default_temperature,
        )
        reasoning, answer = self._split_thinking(raw)
        return {
            "answer": answer.strip(),
            "reasoning": reasoning.strip() if reasoning else None,
            "context_used": bool(context),
            "sources": sources,
        }

    # ── Internals ──────────────────────────────────────────────────────────────
    def _chat(self, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
        self.load()
        import torch

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Only decode the newly generated tokens.
        generated = output_ids[0][inputs.shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    @staticmethod
    def _split_thinking(text: str) -> tuple[str | None, str]:
        """Separate a <think>...</think> reasoning trace from the final answer."""
        import re
        match = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, text

    def _retrieve_context(self, query: str) -> tuple[str, list[dict]]:
        """
        Retrieve the most relevant context across *all* knowledge graphs in the
        attached storage, via the shared (cached) RetrievalService.

        Returns (context_text, sources). Degrades to ("", []) if no KGs or the
        retrieval dependencies are missing, so generation still works.
        """
        try:
            from api.services.retrieval_service import get_retrieval_service

            retrieval = get_retrieval_service(self.config)
            result = retrieval.retrieve(query)
            return result.get("context", ""), result.get("sources", [])
        except Exception:
            return "", []
