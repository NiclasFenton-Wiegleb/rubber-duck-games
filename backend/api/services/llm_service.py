"""
Language Model service.

Loads a HuggingFace chat model (configurable via LLM_MODEL_ID; defaults to
``google/gemma-4-E4B-it``) as a process-wide singleton and serves two kinds of
output:

  - generate_simple():  a direct, fast answer (no extended reasoning). Implements
                        the full "simple request" workstream: file-based system
                        prompt + multi-KG context retrieval + LLM generation.
  - generate_complex(): a richer answer with optional "thinking" mode and,
                        optionally, Knowledge-Graph / RAG context.

The model is heavy, so we keep a process-wide singleton and (by default) load
it lazily on the first request. Transformers/torch are imported lazily so the
Flask app can boot without them installed.

To swap in a different model, set the ``LLM_MODEL_ID`` environment variable to
any HuggingFace model id compatible with ``AutoModelForCausalLM`` and
``AutoProcessor`` (e.g. ``LLM_MODEL_ID=niclasfw/smollm3-3b-codex``).
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("llm_service")

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
        self._processor = None
        self._load_lock = threading.Lock()
        # Resolved at load() time — the concrete device the model actually
        # ended up on (e.g. "cuda" may resolve to "cpu" with no GPU).
        self.device: str | None = None

    # ── Model loading ────────────────────────────────────────────────────────
    def load(self):
        """Load processor + model into memory (idempotent).

        Uses ``AutoProcessor`` (required by Gemma 4 and other recent models
        that ship a ``processor_config.json``) and loads the model with
        ``dtype="auto"`` / ``device_map="auto"`` so accelerate handles
        layer placement and precision automatically. This matches the
        official model-card snippet for ``google/gemma-4-E4B-it``.
        """
        if self._model is not None:
            log.debug("Model already loaded, skipping load()")
            return
        with self._load_lock:
            if self._model is not None:
                log.debug("Model already loaded (double-checked), skipping load()")
                return

            from transformers import AutoModelForCausalLM, AutoProcessor

            log.info("Loading model '%s' (device_pref=%s) ...",
                     self.model_id, self.device_pref)
            t0 = time.time()

            # AutoProcessor handles tokenizer + image processor etc. for
            # multi-modal models. For text-only generation we only need the
            # `.tokenizer` attribute (exposed via the _tokenizer property).
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype="auto",       # let the model pick its native dtype
                device_map="auto",  # let accelerate place layers optimally
            )

            # Record the concrete device the model actually landed on so
            # callers (e.g. the UI status chip) can report it accurately.
            self.device = str(self._model.device)
            log.info("Model loaded successfully in %.1fs (device=%s)",
                     time.time() - t0, self.device)

    @property
    def _tokenizer(self):
        """Return the tokenizer backing the processor.

        ``AutoProcessor`` wraps a tokenizer (among other things like an
        image processor). For text-only generation we only need the tokenizer.
        """
        if self._processor is None:
            return None
        return self._processor.tokenizer

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
        Richer answer: enables reasoning mode ("/think") and optionally
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
        log.info("Generating: input_tokens will be tokenized, max_new=%d, temp=%.3f",
                 max_new_tokens, temperature)
        t0 = time.time()

        self.load()
        import torch

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)

        # apply_chat_template may return a BatchEncoding (dict-like wrapper)
        # instead of a raw tensor.  Use .input_ids when available so .shape
        # does not trigger __getattr__ → KeyError → AttributeError.
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs
        input_len = input_ids.shape[-1]
        log.debug("Tokenized %d input tokens on device '%s'",
                   input_len, self._model.device)

        # Build kwargs for model.generate – pass the raw tensor and, if
        # available, the attention_mask so the model knows what to ignore.
        generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-4),
            "top_p": 0.95,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if hasattr(inputs, "attention_mask"):
            generate_kwargs["attention_mask"] = inputs.attention_mask

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                **generate_kwargs,
            )

        # Only decode the newly generated tokens.
        generated = output_ids[0][input_len:]
        output_len = len(generated)
        decoded = self._tokenizer.decode(generated, skip_special_tokens=True)

        elapsed = time.time() - t0
        log.info("Generation done: input=%d tokens, output=%d tokens, "
                 "elapsed=%.1fs (%.1f tok/s)",
                 input_len, output_len, elapsed,
                 output_len / max(elapsed, 0.001))
        return decoded

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