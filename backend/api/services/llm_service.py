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
any HuggingFace model id compatible with ``AutoModelForCausalLM``.
"""

from __future__ import annotations

import logging
import os
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
        # Load the weights in 4-bit (bitsandbytes NF4) when requested. This
        # roughly quarters VRAM and speeds up generation on a single GPU; it
        # requires CUDA + bitsandbytes and is silently skipped on CPU.
        self.load_in_4bit = bool(config.get("LLM_LOAD_IN_4BIT", False))


        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        # Resolved at load() time — the concrete device the model actually
        # ended up on (e.g. "cuda" may resolve to "cpu" with no GPU).
        self.device: str | None = None

    # ── Model loading ────────────────────────────────────────────────────────
    def load(self):
        """Load tokenizer + model into memory (idempotent).

        This service only does **text-to-text** generation, so we load a
        bare ``AutoTokenizer`` (which carries Gemma 4's chat template) and a
        causal-LM head — no multimodal ``AutoProcessor`` is needed.

        For speed we:
          * load the weights in 4-bit (bitsandbytes NF4) when ``LLM_LOAD_IN_4BIT``
            is set and CUDA is available — this quarters VRAM and is markedly
            faster than fp32 on a single GPU; and
          * otherwise pick an explicit half-precision dtype on GPU
            (``float16``/``bfloat16``) instead of the checkpoint's native (often
            fp32) precision, which is what made T4 inference slow.

        ``device_map="auto"`` still lets accelerate place layers optimally.
        """

        if self._model is not None:
            log.debug("Model already loaded, skipping load()")
            return
        with self._load_lock:
            if self._model is not None:
                log.debug("Model already loaded (double-checked), skipping load()")
                return

            from transformers import AutoModelForCausalLM

            log.info("Loading model '%s' (device_pref=%s, load_in_4bit=%s) ...",
                     self.model_id, self.device_pref, self.load_in_4bit)
            t0 = time.time()

            self._processor = self._load_processor_or_tokenizer()

            model_kwargs: dict = {"device_map": "auto"}
            quant_config = self._build_quantization_config()
            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config
            else:
                # No quantization — at least pick an efficient dtype per device.
                model_kwargs["dtype"] = self._resolve_dtype()

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )

            # Record the concrete device the model actually landed on so
            # callers (e.g. the UI status chip) can report it accurately.
            self.device = str(self._model.device)
            log.info("Model loaded successfully in %.1fs (device=%s, quantized=%s)",
                     time.time() - t0, self.device, quant_config is not None)

    def _cuda_available(self) -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _on_zero_gpu() -> bool:
        """Detect a HuggingFace ZeroGPU runtime.

        ZeroGPU sets ``SPACES_ZERO_GPU`` in the environment. On that runtime the
        physical GPU is attached only inside a ``@spaces.GPU`` worker, which the
        ``spaces`` package spins up by *forking* the main process. bitsandbytes
        must not be used there (see ``_build_quantization_config``).
        """
        return os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true", "yes")

    def _resolve_dtype(self):
        """Pick an explicit dtype: half precision on GPU, ``auto`` on CPU.

        On a T4 the checkpoint's native dtype is often fp32, which is the main
        reason generation was slow. T4 has no bf16 support, so we use fp16
        there and prefer bf16 only on GPUs that report support for it.

        On ZeroGPU CUDA is not visible in the host process (it is attached only
        inside a ``@spaces.GPU`` worker), so ``torch.cuda.is_bf16_supported()``
        would report False here even though the worker GPU supports bf16. We
        therefore prefer bf16 on ZeroGPU explicitly — the ``spaces`` layer packs
        the bf16 tensors at startup and unpacks them onto the real GPU.
        """
        try:
            import torch
        except Exception:
            return "auto"

        if self._on_zero_gpu():
            return torch.bfloat16

        if not torch.cuda.is_available():
            return "auto"  # CPU: let the checkpoint decide (fp32 typically).
        if getattr(torch.cuda, "is_bf16_supported", None) and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _build_quantization_config(self):
        """Return a bitsandbytes 4-bit config, or None if not applicable.

        4-bit needs CUDA + bitsandbytes; on CPU (or if the dependency is
        missing) we silently fall back to a normal half/auto-precision load so
        the app keeps working everywhere.

        On HuggingFace **ZeroGPU** we deliberately skip bitsandbytes entirely.
        bitsandbytes initialises a *real* CUDA context in the host process at
        load time (it bypasses the ``spaces`` torch-CUDA patch with its own
        native CUDA calls). That poisons ZeroGPU's fork-based worker so the very
        first ``@spaces.GPU`` call dies in ``worker_init`` with
        ``RuntimeError: No CUDA GPUs are available``. The ZeroGPU partition has
        plenty of VRAM, so we just load in half precision (bf16/fp16) instead —
        no quantization needed.
        """
        if not self.load_in_4bit:
            return None
        if self._on_zero_gpu():
            log.warning("LLM_LOAD_IN_4BIT set but running on ZeroGPU — skipping "
                        "bitsandbytes (it breaks ZeroGPU's fork worker) and "
                        "loading in half precision instead.")
            return None
        if not self._cuda_available():
            log.warning("LLM_LOAD_IN_4BIT set but CUDA is unavailable — "
                        "loading without quantization.")
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            log.warning("4-bit quantization unavailable (%s) — "
                        "loading without quantization.", exc)
            return None

        compute_dtype = (
            torch.bfloat16
            if getattr(torch.cuda, "is_bf16_supported", None) and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )


    def _load_processor_or_tokenizer(self):
        """Load a bare ``AutoTokenizer`` for **text-only** generation.

        This service only does text-to-text, so we deliberately avoid the
        multimodal ``AutoProcessor``: Gemma 4's processor pulls in image /
        audio processing classes that require optional heavy dependencies
        (e.g. ``torchvision``), which would otherwise fail at import with
        ``ModuleNotFoundError: No module named 'torchvision'``. The tokenizer
        already carries the chat template we use via ``apply_chat_template``,
        so it's all we need for text generation.

        Requires ``transformers >= 5.5.0`` — older versions don't know
        Gemma 4 and crash with ``'list' object has no attribute 'keys'``.
        """
        import transformers
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            log.info("Loaded tokenizer via AutoTokenizer (text-only)")
            return tokenizer
        except AttributeError as exc:
            # Typical symptom of a too-old transformers for this checkpoint.
            raise RuntimeError(
                f"Failed to load a tokenizer for '{self.model_id}'. "
                f"The installed transformers "
                f"({getattr(transformers, '__version__', '?')}) is too old for "
                f"this model — Gemma 4 requires transformers>=5.5.0. "
                f"Please upgrade transformers (see requirements.txt)."
            ) from exc



    @property
    def _tokenizer(self):
        """Return the tokenizer.

        When ``self._processor`` is an ``AutoProcessor`` instance the
        tokenizer lives at ``.tokenizer``. When it's a bare tokenizer
        (``AutoTokenizer`` fallback) we return it directly.
        """
        if self._processor is None:
            return None
        return getattr(self._processor, "tokenizer", self._processor)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Public generation methods ─────────────────────────────────────────────

    def generate_simple(self, query: str, max_new_tokens=None, temperature=None,
                         use_context=True, stop_on_json=False) -> dict:
        """
        Direct, fast answer — no extended reasoning.

        Implements the full "simple request" workstream:
          1. system prompt is loaded from a local text file (cached),
          2. the most relevant context across all KGs in the storage bucket is
             retrieved and injected (skippable via use_context=False),
          3. the assembled prompt is run straight through the SLM,
          4. the answer + provenance is returned for the frontend.

        When ``stop_on_json`` is True, generation halts as soon as the output
        forms a complete balanced JSON object, so we don't burn tokens (and
        wall-clock time) padding out to the full ``max_new_tokens`` budget.
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
            stop_on_json=stop_on_json,
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
    def _chat(self, messages: list[dict], max_new_tokens: int, temperature: float,
              stop_on_json: bool = False) -> str:
        log.info("Generating: input_tokens will be tokenized, max_new=%d, temp=%.3f, "
                 "stop_on_json=%s", max_new_tokens, temperature, stop_on_json)
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

        # Stop as soon as a complete JSON object has been emitted so we don't
        # waste tokens/time padding to the full budget (and so the answer is a
        # complete, parseable object rather than a truncated one).
        if stop_on_json:
            criteria = self._json_stop_criteria(input_len)
            if criteria is not None:
                generate_kwargs["stopping_criteria"] = criteria

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

    def _json_stop_criteria(self, prompt_len: int):
        """Build a ``StoppingCriteriaList`` that halts on a complete JSON object.

        The criterion only does the (cheap) brace-balance scan when the most
        recently generated token actually contains a ``}``, so it adds
        negligible overhead per step. Returns ``None`` if the transformers
        stopping-criteria API isn't importable, so generation still works.
        """
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:  # noqa: BLE001
            return None

        tokenizer = self._tokenizer

        class _BalancedJSONStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
                last_id = int(input_ids[0, -1].item())
                piece = tokenizer.decode([last_id], skip_special_tokens=True)
                if "}" not in piece:
                    return False
                text = tokenizer.decode(input_ids[0][prompt_len:],
                                        skip_special_tokens=True)
                start = text.find("{")
                if start < 0:
                    return False
                depth = 0
                in_str = False
                esc = False
                for ch in text[start:]:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return True  # top-level object closed → stop.
                return False

        return StoppingCriteriaList([_BalancedJSONStop()])

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
