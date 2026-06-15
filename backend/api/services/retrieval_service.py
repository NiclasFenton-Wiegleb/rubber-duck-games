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
import pickle
import re
import threading
import time
from collections import Counter
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

        # Also track the output dir so locally-built KGs are found even when
        # KG_STORAGE_DIR points somewhere else (e.g. /data on HF Spaces).
        output = _cfg(config, "KG_OUTPUT_DIR")
        self.output_dir = Path(output) if output else None

        self.embed_model_name = _cfg(config, "EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        # Pin the embedder to an explicit device. On a ZeroGPU Space the host
        # process must NEVER create a CUDA context (it forks the @spaces.GPU
        # workers; a CUDA-initialised host makes them die in ``worker_init``
        # with "No CUDA GPUs are available"). With no device argument,
        # SentenceTransformer auto-selects "cuda" whenever ``torch.cuda
        # .is_available()`` is truthy — and the ``spaces`` patch makes that
        # report True in the host — which would initialise a real CUDA context
        # here. EMBED_DEVICE defaults to "cpu" for exactly this reason (the
        # embedder is tiny, so CPU is plenty fast); override on dedicated-GPU
        # Spaces if desired.
        self.embed_device = _cfg(config, "EMBED_DEVICE", "cpu")

        self.per_kg_k = int(_cfg(config, "RETRIEVAL_PER_KG_K", 8))
        self.top_k = int(_cfg(config, "RETRIEVAL_TOP_K", 4))
        # Context-size / relevance guards (see config.py for rationale). 0 means
        # "disabled" for the char caps; the score gate is skipped when <= 0.
        self.max_chunk_chars = int(_cfg(config, "RETRIEVAL_MAX_CHUNK_CHARS", 0))
        self.max_context_chars = int(_cfg(config, "RETRIEVAL_MAX_CONTEXT_CHARS", 0))
        self.min_score = float(_cfg(config, "RETRIEVAL_MIN_SCORE", 0.0))

        # Knowledge-graph augmentation knobs (see config.py for rationale). When
        # enabled, each KG's NetworkX graph + entity→chunk map are loaded into
        # RAM at startup and used to expand/re-rank FAISS hits. Pure in-memory
        # dict/graph lookups — no model, no CUDA — so query latency stays flat.
        self.kg_expansion_enabled = bool(_cfg(config, "KG_EXPANSION_ENABLED", True))
        self.kg_expansion_max_neighbors = int(
            _cfg(config, "KG_EXPANSION_MAX_NEIGHBORS", 5)
        )
        self.kg_entity_boost = float(_cfg(config, "KG_ENTITY_BOOST", 0.05))
        self.kg_entity_boost_cap = float(_cfg(config, "KG_ENTITY_BOOST_CAP", 0.20))

        self._embedder = None

        self._kgs: list[dict] | None = None  # cached loaded KGs
        self._load_lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        kg_names: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> dict:
        """
        Return the merged top-k context across selected KGs.

        Result: {
            "context": "…merged snippets…",
            "sources": [{"kg": str, "chunk_id": int, "score": float}, …],
        }
        Always returns successfully; an empty context means no KGs / no deps.
        """
        top_k = top_k or self.top_k
        requested = {name for name in (kg_names or []) if name}
        try:
            self._ensure_loaded()
        except Exception as exc:
            log.warning("Could not load retrieval indexes: %s", exc)
            return {"context": "", "sources": []}

        kgs = [
            kg for kg in (self._kgs or [])
            if not requested or kg["name"] in requested
        ]
        if requested and not kgs:
            log.warning(
                "Retrieval: requested KG(s) %s not loaded; available=%s",
                sorted(requested),
                [kg["name"] for kg in (self._kgs or [])],
            )

        if not kgs:
            log.debug("No knowledge graphs available for retrieval")
            return {"context": "", "sources": []}

        lexical_seeds = self._lexical_seeds(query, kgs)

        try:
            t0 = time.time()
            q_emb = self._embed_query(query)
            log.debug("Query embedding took %.3fs", time.time() - t0)
        except Exception as exc:
            log.warning("Could not embed query: %s", exc)
            return self._context_from_hits(lexical_seeds[:top_k])

        # ── Semantic seeds (FAISS, across every KG) ──────────────────────────
        seeds: list[dict] = []
        for kg in kgs:
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
                seeds.append({
                    "score": float(score),
                    "kg": kg["name"],
                    "kg_obj": kg,
                    "chunk_id": int(cid),
                    "text": text,
                    "expanded": False,
                })

        if lexical_seeds:
            seen = {(h["kg"], h["chunk_id"]) for h in seeds}
            for hit in lexical_seeds:
                key = (hit["kg"], hit["chunk_id"])
                if key in seen:
                    for seed in seeds:
                        if (seed["kg"], seed["chunk_id"]) == key:
                            seed["score"] += 0.35
                            break
                    continue
                seeds.append(hit)
                seen.add(key)

        seeds.sort(key=lambda h: h["score"], reverse=True)

        # Relevance gate: if the best *semantic* hit is too weak, the KGs simply
        # don't cover this query — inject nothing rather than pad the prompt (and
        # slow generation) with irrelevant text. Applied to the FAISS seeds only
        # so graph-expanded neighbours can never bypass it.
        if self.min_score > 0 and (not seeds or seeds[0]["score"] < self.min_score):
            best = seeds[0]["score"] if seeds else float("nan")
            log.info("Retrieval: best score %.4f < min_score %.4f — dropping "
                     "context for this query", best, self.min_score)
            return {"context": "", "sources": []}

        # Drop individual sub-threshold seeds.
        if self.min_score > 0:
            seeds = [h for h in seeds if h["score"] >= self.min_score]

        # ── Knowledge-graph augmentation (expand + re-rank) ──────────────────
        # Uses each KG's in-memory graph/entity maps to pull entity-linked
        # neighbour chunks the vector search missed and to boost chunks that are
        # well-connected to the other seeds. Pure dict/graph lookups, so the
        # query hot-path stays fast.
        candidates = (
            self._augment_with_graph(seeds)
            if self.kg_expansion_enabled
            else seeds
        )
        candidates.sort(key=lambda h: h["score"], reverse=True)
        top = candidates[:top_k]

        return self._context_from_hits(top)

    def _context_from_hits(self, hits: list[dict]) -> dict:
        # Trim each chunk and the merged whole so input length (and therefore
        # prefill time) stays bounded regardless of chunk size.
        parts: list[str] = []
        sources: list[dict] = []
        total = 0
        for h in hits:
            text = self._truncate(h["text"], self.max_chunk_chars)
            snippet = f"[{h['kg']}] {text}"
            if self.max_context_chars > 0 and total + len(snippet) > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining <= 0:
                    break
                snippet = snippet[:remaining]
                parts.append(snippet)
                sources.append({
                    "kg": h["kg"], "chunk_id": h["chunk_id"],
                    "score": round(h["score"], 4),
                })
                break
            parts.append(snippet)
            sources.append({
                "kg": h["kg"], "chunk_id": h["chunk_id"],
                "score": round(h["score"], 4),
            })
            total += len(snippet) + 2  # account for the "\n\n" joiner

        context = "\n\n".join(parts)
        return {"context": context, "sources": sources}

    def _lexical_seeds(self, query: str, kgs: list[dict]) -> list[dict]:
        """Return high-confidence hits for exact symbols mentioned in the query."""
        terms = set(re.findall(r"`([^`]+)`", query or ""))
        terms.update(re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", query or ""))
        identifiers = {
            term.strip()
            for term in terms
            if term.strip() and any(ch in term for ch in ("_", "/", "."))
        }
        if not identifiers:
            return []

        hits: list[dict] = []
        for kg in kgs:
            for chunk_id, text in kg["chunks"].items():
                lower_text = text.lower()
                matches = sum(1 for term in identifiers if term.lower() in lower_text)
                if not matches:
                    continue
                hits.append({
                    "score": 1.0 + (matches * 0.2),
                    "kg": kg["name"],
                    "kg_obj": kg,
                    "chunk_id": int(chunk_id),
                    "text": text,
                    "expanded": False,
                })
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Cut text to at most ``max_chars`` (0 disables truncation)."""
        if max_chars and max_chars > 0 and len(text) > max_chars:
            return text[:max_chars].rstrip() + " …"
        return text


    def refresh(self) -> int:
        """Drop cached indexes so the next query reloads from disk. Returns KG count."""
        with self._load_lock:
            self._kgs = None
        self._ensure_loaded()
        count = len(self._kgs or [])
        log.info("Retrieval cache refreshed — %d KG(s) available: %s",
                 count, self.kg_names)
        return count

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
                    # Pass an explicit device so SentenceTransformer never
                    # auto-selects "cuda" and initialises a real CUDA context
                    # (fatal in a ZeroGPU host — see __init__ for details).
                    log.info("Retrieval: loading embedder '%s' on device=%s",
                             self.embed_model_name, self.embed_device)
                    self._embedder = SentenceTransformer(
                        self.embed_model_name, device=self.embed_device
                    )
        return self._embedder


    def _ensure_loaded(self):
        if self._kgs is not None:
            return
        with self._load_lock:
            if self._kgs is not None:
                return
            import faiss  # lazy

            kg_dirs = self._discover_kg_dirs()
            log.info("Retrieval: discovered %d KG director(ies) in %s: %s",
                     len(kg_dirs), self.storage_dir,
                     [d.name for d in kg_dirs] if kg_dirs else "(none)")

            kgs: list[dict] = []
            for kg_dir in kg_dirs:
                index_path = kg_dir / "faiss" / "index.faiss"
                chunks_path = kg_dir / "chunks" / "chunks.jsonl"
                try:
                    index = faiss.read_index(str(index_path))
                    chunk_text = self._load_chunk_texts(chunks_path)
                except Exception as exc:
                    log.warning("Retrieval: failed to load KG '%s': %s",
                                kg_dir.name, exc)
                    continue
                if not chunk_text:
                    log.warning("Retrieval: no chunks found for KG '%s'", kg_dir.name)
                    continue

                # Pull the knowledge-graph assets into RAM alongside the index so
                # query-time augmentation never touches disk. Loading is fully
                # optional: a KG with no graph.pkl / entity map degrades to pure
                # vector search. Nothing here touches CUDA (plain pickle/json), so
                # it is safe to run in the ZeroGPU host that forks GPU workers.
                graph, entity_to_chunks, chunk_entities = self._load_graph_assets(kg_dir)

                kgs.append({
                    "name": kg_dir.name if kg_dir != self.storage_dir else "default",
                    "index": index,
                    "chunks": chunk_text,
                    "graph": graph,
                    "entity_to_chunks": entity_to_chunks,
                    "chunk_entities": chunk_entities,
                })
                log.info(
                    "Retrieval: loaded KG '%s' (%d chunks, %d entities, "
                    "graph=%s)",
                    kgs[-1]["name"], len(chunk_text), len(entity_to_chunks),
                    "yes" if graph is not None else "no",
                )
            self._kgs = kgs
            log.info("Retrieval: %d KG(s) loaded into memory", len(kgs))


    def _discover_kg_dirs(self) -> list[Path]:
        """Find every KG folder under the storage root and (if different) the
        local output dir.  This makes locally-built KGs immediately available
        for RAG queries even when a storage bucket is also attached."""
        roots: list[Path] = [self.storage_dir]
        if self.output_dir is not None and self.output_dir != self.storage_dir:
            roots.append(self.output_dir)

        found: list[Path] = []
        seen: set[str] = set()  # dedupe by KG name
        for root in roots:
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "faiss" / "index.faiss").exists() and \
                   (child / "chunks" / "chunks.jsonl").exists():
                    name = child.name
                    if name not in seen:
                        found.append(child)
                        seen.add(name)
            # Fall back to a flat single-KG layout at the root itself.
            if (root / "faiss" / "index.faiss").exists() and \
               (root / "chunks" / "chunks.jsonl").exists():
                if "default" not in seen:
                    found.append(root)
                    seen.add("default")
        return found

    # ── Knowledge-graph augmentation ──────────────────────────────────────────
    @staticmethod
    def _load_graph_assets(kg_dir: Path):
        """Load a KG's NetworkX graph + entity→chunk map into RAM.

        Returns ``(graph, entity_to_chunks, chunk_entities)`` where:
          * ``graph``            – the unpickled NetworkX graph (or ``None``),
          * ``entity_to_chunks`` – ``{entity_name: [chunk_id, …]}`` (or ``{}``),
          * ``chunk_entities``   – the inverse ``{chunk_id: {entity_name, …}}``.

        Everything is optional and best-effort: a KG missing these files simply
        falls back to pure vector search. No CUDA / model code runs here — only
        ``pickle`` + ``json`` — so it is safe in the ZeroGPU host process.
        """
        graph = None
        graph_path = kg_dir / "kg" / "graph.pkl"
        if graph_path.exists():
            try:
                with open(graph_path, "rb") as f:
                    graph = pickle.load(f)
            except Exception as exc:  # noqa: BLE001
                log.warning("Retrieval: failed to load graph for KG '%s': %s",
                            kg_dir.name, exc)
                graph = None

        entity_to_chunks: dict[str, list[int]] = {}
        e2c_path = kg_dir / "kg" / "entity_to_chunks.json"
        if e2c_path.exists():
            try:
                with open(e2c_path, encoding="utf-8") as f:
                    raw = json.load(f)
                # Normalise chunk ids to ints; keys stay entity-name strings.
                entity_to_chunks = {
                    str(ent): [int(c) for c in cids]
                    for ent, cids in raw.items()
                }
            except Exception as exc:  # noqa: BLE001
                log.warning("Retrieval: failed to load entity map for KG '%s': %s",
                            kg_dir.name, exc)
                entity_to_chunks = {}

        # Invert entity→chunks into chunk→entities so the query path can look up
        # a chunk's entities in O(1).
        chunk_entities: dict[int, set[str]] = {}
        for ent, cids in entity_to_chunks.items():
            for cid in cids:
                chunk_entities.setdefault(cid, set()).add(ent)

        return graph, entity_to_chunks, chunk_entities

    @staticmethod
    def _chunk_entities(kg: dict, chunk_id: int) -> set[str]:
        """Return the entity names linked to a chunk (empty set if unknown)."""
        return kg.get("chunk_entities", {}).get(chunk_id, set())

    def _neighbour_chunks(self, kg: dict, entities: set[str],
                          exclude: int, limit: int) -> list[int]:
        """Find chunks that share entities with a seed, ranked by overlap.

        Walks the entity→chunk side of the graph: each of the seed's entities
        points at the other chunks that mention it, and chunks mentioned by more
        of the seed's entities rank higher. Returns at most ``limit`` chunk ids.
        """
        e2c = kg.get("entity_to_chunks") or {}
        counts: Counter[int] = Counter()
        for ent in entities:
            for cid in e2c.get(ent, ()):  # entity → mentioning chunks
                if cid == exclude:
                    continue
                counts[cid] += 1
        return [cid for cid, _ in counts.most_common(limit)]

    def _augment_with_graph(self, seeds: list[dict]) -> list[dict]:
        """Expand the FAISS seeds with entity-linked neighbours and re-rank.

        1. **Expand** – for each seed chunk, pull the most strongly entity-linked
           neighbour chunks (within the same KG) that the vector search missed.
        2. **Re-rank** – boost any candidate that shares entities with the other
           seeds, so chunks reinforced by the graph's entity structure rise.

        All operations are in-memory dict/Counter lookups over the maps loaded at
        startup, so this adds negligible latency to the query hot-path.
        """
        if not seeds:
            return seeds

        # Entity set of every seed chunk (used for expansion + re-ranking).
        seed_entities: dict[int, set[str]] = {
            idx: self._chunk_entities(s["kg_obj"], s["chunk_id"])
            for idx, s in enumerate(seeds)
        }

        # Candidate pool keyed by (kg-name, chunk_id) so seeds and neighbours
        # dedupe cleanly; seeds always win over a weaker expanded duplicate.
        by_key: dict[tuple[str, int], dict] = {
            (s["kg"], s["chunk_id"]): s for s in seeds
        }

        # ── 1. Expansion ─────────────────────────────────────────────────────
        if self.kg_expansion_max_neighbors > 0:
            for idx, s in enumerate(seeds):
                kg = s["kg_obj"]
                ents = seed_entities[idx]
                if not ents:
                    continue
                for cid in self._neighbour_chunks(
                    kg, ents, exclude=s["chunk_id"],
                    limit=self.kg_expansion_max_neighbors,
                ):
                    key = (kg["name"], cid)
                    if key in by_key:
                        continue  # already a (stronger) candidate
                    text = kg["chunks"].get(cid)
                    if not text:
                        continue
                    by_key[key] = {
                        # Neighbours are entity-linked, not directly matched, so
                        # they inherit a decayed fraction of their seed's score —
                        # ranking below the seed but able to surface when the
                        # semantic hits are sparse.
                        "score": s["score"] * 0.5,
                        "kg": kg["name"],
                        "kg_obj": kg,
                        "chunk_id": cid,
                        "text": text,
                        "expanded": True,
                    }

        # ── 2. Entity re-ranking ─────────────────────────────────────────────
        if self.kg_entity_boost > 0:
            # How many seed chunks each entity appears in.
            seed_freq: Counter[str] = Counter()
            for ents in seed_entities.values():
                seed_freq.update(ents)

            for cand in by_key.values():
                cand_ents = self._chunk_entities(cand["kg_obj"], cand["chunk_id"])
                if not cand_ents:
                    continue
                # Count entities this candidate shares with *other* seeds. A
                # non-expanded seed already contributes 1 to seed_freq for each
                # of its own entities, so discount that self-count.
                own = 0 if cand["expanded"] else 1
                shared = sum(
                    1 for e in cand_ents if (seed_freq.get(e, 0) - own) >= 1
                )
                if shared <= 0:
                    continue
                bonus = min(shared * self.kg_entity_boost, self.kg_entity_boost_cap)
                cand["score"] += bonus

        return list(by_key.values())

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
                    source = rec.get("source_file") or rec.get("doc_title") or "unknown"
                    mapping[int(rec["chunk_id"])] = (
                        f"Source file: {source}\n{rec['text']}"
                    )
                except (ValueError, KeyError):
                    continue
        return mapping
