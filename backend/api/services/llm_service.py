"""
Small Language Model service.

Loads the locally hosted, fine-tuned SmolLM3-3B model
(`niclasfw/smollm3-3b-codex` by default) once and serves two kinds of output:

  - generate_simple():  a direct, fast answer (no extended reasoning).
  - generate_complex(): a richer answer using SmolLM3's "thinking" mode and,
                        optionally, Knowledge-Graph / RAG context retrieved from
                        the artifacts produced by KnowledgeGraphService.

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

    # ── Model loading ────────────────────────────────────────────────────────
    def load(self):
        """Load tokenizer + model into memory (idempotent)."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device_map = "auto"
            if self.device_pref in ("cpu", "cuda"):
                device_map = {"": self.device_pref}

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Public generation methods ─────────────────────────────────────────────
    def generate_simple(self, query: str, max_new_tokens=None, temperature=None) -> dict:
        """Direct answer — no extended reasoning."""
        messages = [
            {"role": "system", "content": "You are a concise, helpful coding assistant. /no_think"},
            {"role": "user", "content": query},
        ]
        answer = self._chat(
            messages,
            max_new_tokens=max_new_tokens or self.default_max_new,
            temperature=temperature if temperature is not None else self.default_temperature,
        )
        return {"answer": answer.strip(), "reasoning": None}

    def generate_complex(self, query: str, use_context=True,
                         max_new_tokens=None, temperature=None) -> dict:
        """
        Richer answer: enables SmolLM3 reasoning ("/think") and optionally
        injects Knowledge-Graph / RAG context. Splits out the model's
        <think>...</think> trace into a separate `reasoning` field.
        """
        context = self._retrieve_context(query) if use_context else ""
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

    def _retrieve_context(self, query: str, top_k: int = 4) -> str:
        """
        Best-effort RAG retrieval from locally built KG/FAISS artifacts.

        Returns an empty string if the artifacts or dependencies are missing,
        so the complex endpoint still works without a built knowledge graph.
        """
        try:
            import json
            from pathlib import Path

            import faiss
            from sentence_transformers import SentenceTransformer

            out_dir = Path(self.config["KG_OUTPUT_DIR"])
            index_path = out_dir / "faiss" / "index.faiss"
            chunks_path = out_dir / "chunks" / "chunks.jsonl"
            if not index_path.exists() or not chunks_path.exists():
                return ""

            index = faiss.read_index(str(index_path))
            embedder = SentenceTransformer(self.config["EMBED_MODEL"])
            q_emb = embedder.encode(
                [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
            )
            _, ids = index.search(q_emb, top_k)

            wanted = {int(i) for i in ids[0] if i != -1}
            texts = []
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["chunk_id"] in wanted:
                        texts.append(f"- {rec['text']}")
            return "\n".join(texts)
        except Exception:
            return ""
