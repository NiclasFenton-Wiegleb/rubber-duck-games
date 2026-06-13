"""
Retrieval service.

Pulls the most relevant chunks for a user query across *all* knowledge graphs
present in the attached storage, and returns them as a single merged context
block ready to inject into a prompt.

Storage layout (driven by KG_STORAGE_DIR):
    <root>/<kg-name>/faiss/index.faiss
    <root>/<kg-name>/chunks/chunks.jsonl
    <root>/<kg-name>/manifest.json        (optional)

On a HuggingFace Space the storage bucket is mounted under KG_STORAGE_DIR
(e.g. "/data") with one subfolder per knowledge graph. For local development we
fall back to KG_OUTPUT_DIR. A *flat* single-KG layout (<root>/faiss/index.faiss)
is also supported transparently.

Speed is a priority, so:
  - the SentenceTransformer embedder is loaded once (process-wide singleton),
  - every FAISS index + its chunk-id→text map is loaded once and kept in RAM,
  - the query is embedded a single time and reused across all KGs.

Heavy deps (faiss, sentence-transformers) are imported lazily so the Flask app
can boot without them installed; if they're missing, retrieval degrades to an
empty context instead of crashing the request.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("retrieval_service")

# Process-wide singleton — the embedder + indexes are expensive to load.
_INSTANCE: "RetrievalService | None" = None
_LOCK = threading.Lock()


def get_retrieval_service(config) -> "RetrievalService":
    """Return the shared RetrievalService, creating it on first use."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = RetrievalService(config)
    return _INSTANCE


def _cfg(config, key, default=None):
    """Read a key from a Flask config (dict-like) or plain dict, with default."""
    try:
        value = config.get(key, default)
    except AttributeError:
        value = config[key] if key in config else default
    return value if value is not None else default


class RetrievalService:
    def __init__(self, config):
        self.config = config

        storage = _cfg(config, "KG_STORAGE_DIR") or _cfg(config, "KG_OUTPUT_DIR")
        if not storage:
            storage = Path(__file__).resolve().parents[2] / "artifacts"
        self.storage_dir = Path(storage)

        self.embed_model_name = _cfg(config, "EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        self.per_kg_k = int(_cfg(config, "RETRIEVAL_PER_KG_K", 8))
        self.top_k = int(_cfg(config, "RETRIEVAL_TOP_K", 4))

        self._embedder = None
        self._kgs: list[dict] | None = None  # cached loaded KGs
        self._load_lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int | None = None) -> dict:
        """
        Return the merged top-k context across all KGs.

        Result: {
            "context": "…merged snippets…",
            "sources": [{"kg": str, "chunk_id": int, "score": float}, …],
        }
        Always returns successfully; an empty context means no KGs / no deps.
        """
        top_k = top_k or self.top_k
        try:
            self._ensure_loaded()
        except Exception as exc:
            log.warning("Could not load retrieval indexes: %s", exc)
            return {"context": "", "sources": []}

        if not self._kgs:
            log.debug("No knowledge graphs available for retrieval")
            return {"context": "", "sources": []}

        try:
            t0 = time.time()
            q_emb = self._embed_query(query)
            log.debug("Query embedding took %.3fs", time.time() - t0)
        except Exception as exc:
            log.warning("Could not embed query: %s", exc)
            return {"context": "", "sources": []}

        hits: list[dict] = []
        for kg in self._kgs:
            try:
                scores, ids = kg["index"].search(q_emb, self.per_kg_k)
            except Exception:
                continue
            for score, cid in zip(scores[0], ids[0]):
                if cid == -1:
                    continue
                text = kg["chunks"].get(int(cid))
                if not text:
                    continue
                hits.append({
                    "score": float(score),
                    "kg": kg["name"],
                    "chunk_id": int(cid),
                    "text": text,
                })

        hits.sort(key=lambda h: h["score"], reverse=True)
        top = hits[:top_k]

        context = "\n\n".join(f"[{h['kg']}] {h['text']}" for h in top)
        sources = [
            {"kg": h["kg"], "chunk_id": h["chunk_id"], "score": round(h["score"], 4)}
            for h in top
        ]
        return {"context": context, "sources": sources}

    def refresh(self) -> int:
        """Drop cached indexes so the next query reloads from disk. Returns KG count."""
        with self._load_lock:
            self._kgs = None
        self._ensure_loaded()
        return len(self._kgs or [])

    @property
    def kg_names(self) -> list[str]:
        try:
            self._ensure_loaded()
        except Exception:
            return []
        return [kg["name"] for kg in (self._kgs or [])]

    # ── Internals ────────────────────────────────────────────────────────────
    def _embed_query(self, query: str):
        embedder = self._embedder_instance()
        return embedder.encode(
            [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
        )

    def _embedder_instance(self):
        if self._embedder is None:
            with self._load_lock:
                if self._embedder is None:
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _ensure_loaded(self):
        if self._kgs is not None:
            return
        with self._load_lock:
            if self._kgs is not None:
                return
            import faiss  # lazy

            kgs: list[dict] = []
            for kg_dir in self._discover_kg_dirs():
                index_path = kg_dir / "faiss" / "index.faiss"
                chunks_path = kg_dir / "chunks" / "chunks.jsonl"
                try:
                    index = faiss.read_index(str(index_path))
                    chunk_text = self._load_chunk_texts(chunks_path)
                except Exception:
                    continue
                if not chunk_text:
                    continue
                kgs.append({
                    "name": kg_dir.name if kg_dir != self.storage_dir else "default",
                    "index": index,
                    "chunks": chunk_text,
                })
            self._kgs = kgs

    def _discover_kg_dirs(self) -> list[Path]:
        """Find every KG folder under the storage root (subfolders or flat)."""
        root = self.storage_dir
        if not root.exists():
            return []

        found: list[Path] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "faiss" / "index.faiss").exists() and \
               (child / "chunks" / "chunks.jsonl").exists():
                found.append(child)

        # Fall back to a flat single-KG layout at the root itself.
        if not found and (root / "faiss" / "index.faiss").exists() and \
           (root / "chunks" / "chunks.jsonl").exists():
            found.append(root)

        return found

    @staticmethod
    def _load_chunk_texts(chunks_path: Path) -> dict[int, str]:
        mapping: dict[int, str] = {}
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    mapping[int(rec["chunk_id"])] = rec["text"]
                except (ValueError, KeyError):
                    continue
        return mapping
