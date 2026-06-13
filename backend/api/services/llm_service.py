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
``AutoTokenizer`` (e.g. ``LLM_MODEL_ID=niclasfw/smollm3-3b-codex``).
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
        self._tokenizer = None
        self._load_lock = threading.Lock()
        # Resolved at load() time — the concrete device the model actually
        # ended up on (e.g. "cuda" pref may resolve to "cpu" with no GPU).
        self.device: str | None = None


    # ── Model loading ────────────────────────────────────────────────────────
    def load(self):
        """Load tokenizer + model into memory (idempotent)."""
        if self._model is not None:
            log.debug("Model already loaded, skipping load()")
            return
        with self._load_lock:
            if self._model is not None:
                log.debug("Model already loaded (double-checked), skipping load()")
                return
            import torch
            from transformers import AutoModelForCausalLM

            device = self._resolve_device(torch)
            log.info("Loading model '%s' on device '%s' (pref=%s) ...",
                     self.model_id, device, self.device_pref)
            t0 = time.time()

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
            log.info("Model loaded successfully in %.1fs (device=%s, dtype=%s)",
                     time.time() - t0, device, dtype)

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

        Handles several known compatibility issues between checkpoint metadata
        and the installed ``transformers`` version:

        1. Some checkpoints (e.g. those saved with a bleeding-edge / dev build
           of ``transformers``) write ``"tokenizer_class": "TokenizersBackend"``
           into ``tokenizer_config.json``.  Older ``transformers`` releases
           don't know that class, so ``AutoTokenizer`` raises::

               ValueError: Tokenizer class TokenizersBackend does not exist …

        2. Gemma 4 (and similar recent checkpoints) ship a
           ``tokenizer_config.json`` whose ``added_tokens_decoder`` key is an
           **empty list** (``[]``) instead of an empty dict (``{}``).  Older
           ``transformers`` versions fail inside
           ``_set_model_specific_special_tokens`` with::

               AttributeError: 'list' object has no attribute 'keys'

        3. Some checkpoints store the chat template under ``chat_template``
           (singular, as a string) while others use ``chat_templates``
           (plural, as a dict).  We normalise to ``chat_template``.

        Regardless of the error, we fall back to downloading and patching the
        tokenizer config, then loading the fast tokenizer from the patched
        config.
        """
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(self.model_id)
        except Exception:
            log.warning(
                "AutoTokenizer.from_pretrained failed for '%s' – "
                "falling back to patched tokenizer config.",
                self.model_id, exc_info=True,
            )
            return self._load_tokenizer_with_fixed_config()

    def _load_tokenizer_with_fixed_config(self):
        """Download the tokenizer config, patch known incompatibilities,
        and load a :class:`~transformers.PreTrainedTokenizerFast` from the
        patched files."""
        from pathlib import Path

        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import PreTrainedTokenizerFast

        tmp_dir = None
        try:
            # Download all tokenizer files into a temporary directory so the
            # fast tokenizer can find tokenizer.json, chat_template.jinja, etc.
            log.info("Downloading tokenizer files for '%s' …", self.model_id)
            tmp_dir = Path(snapshot_download(
                self.model_id,
                allow_patterns=[
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                ],
            ))

            config_path = tmp_dir / "tokenizer_config.json"
            if not config_path.exists():
                # snapshot_download may have placed it elsewhere; try direct
                config_path = Path(hf_hub_download(
                    self.model_id, "tokenizer_config.json",
                ))
                tmp_dir = config_path.parent

            self._patch_tokenizer_config(config_path)

            log.info(
                "Loading fast tokenizer from patched config at %s",
                tmp_dir,
            )
            return PreTrainedTokenizerFast.from_pretrained(str(tmp_dir))

        except Exception:
            log.exception(
                "Failed to load tokenizer for '%s' even with patched config.",
                self.model_id,
            )
            raise

    @staticmethod
    def _patch_tokenizer_config(config_path):
        """Patch known incompatibilities in ``tokenizer_config.json`` *in place*.

        Patches applied:

        * ``added_tokens_decoder``: if it is a list, convert to a dict (Gemma 4
          ships an empty list, but transformers expects a dict).
        * ``chat_template``: normalise to a string (some checkpoints use a
          single-item dict or an array, which breaks the jinja renderer).
        """
        import json

        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

        patched = False

        # ── added_tokens_decoder: list → dict ──────────────────────────
        atd = cfg.get("added_tokens_decoder")
        if isinstance(atd, list) and not isinstance(atd, dict):
            # Convert list to dict keyed by token id (if items are dicts
            # with an "id" field) or just use enumerated integer keys.
            try:
                new_atd = {}
                for item in atd:
                    tid = item["id"] if isinstance(item, dict) else len(new_atd)
                    new_atd[tid] = item
                cfg["added_tokens_decoder"] = new_atd
                patched = True
                log.info(
                    "Patched added_tokens_decoder: list(%d) → dict(%d)",
                    len(atd), len(new_atd),
                )
            except Exception:
                # If conversion fails, replace with empty dict and let
                # transformers handle missing entries gracefully.
                cfg["added_tokens_decoder"] = {}
                patched = True
                log.warning("Patched added_tokens_decoder: list → {} (fallback)")

        # ── chat_template: normalise to string ─────────────────────────
        ct = cfg.get("chat_template")
        if ct is not None and not isinstance(ct, str):
            if isinstance(ct, dict) and len(ct) == 1:
                cfg["chat_template"] = list(ct.values())[0]
            elif isinstance(ct, list):
                cfg["chat_template"] = ct[0] if ct else ""
            else:
                cfg["chat_template"] = str(ct)
            patched = True
            log.info("Patched chat_template: %s → str", type(ct).__name__)

        if patched:
            # Write the patched config back so PreTrainedTokenizerFast can
            # read it from disk.
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2, ensure_ascii=False)
            log.info("Patched tokenizer_config.json written to %s", config_path)
        else:
            log.debug("tokenizer_config.json looks fine — no patches applied")

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
