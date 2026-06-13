"""
Knowledge Graph service.

Walks a designated folder, parses every text-like file, chunks the content,
builds a NetworkX knowledge graph (doc / section / chunk / entity nodes),
embeds the chunks into a FAISS index, writes a manifest, and (optionally)
uploads all artifacts to HuggingFace storage.

By default the artifacts are pushed to a HuggingFace **Storage Bucket**
(huggingface.co/buckets/<id>). Set HF_STORAGE_BACKEND=repo to instead commit
them into the Space/dataset git repo (Files tab).

This generalises the pipeline from RubberDuckGames.ipynb so it works on any
folder of files rather than only Sphinx-built Godot HTML.

Heavy dependencies (spacy, sentence-transformers, faiss, networkx,
huggingface_hub) are imported lazily inside methods so the Flask app can boot
even if they are not yet installed.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("kg_service")

# File extensions we treat as ingestible text.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".html", ".htm",
    ".py", ".gd", ".js", ".ts", ".json", ".yaml", ".yml",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".java", ".go", ".rs",
    ".cfg", ".ini", ".toml",
}

# PascalCase / @keyword / $NodeName style entity patterns (from the notebook).
ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z0-9]{2,})\b"        # PascalCase (Node2D, KinematicBody...)
    r"|(@[a-z_]+)"                         # @export, @onready, ...
    r"|(\$[A-Za-z_][A-Za-z0-9_]*)"        # $NodeName shorthand
)

STOP_PASCAL = {
    "The", "This", "That", "With", "When", "For", "You",
    "See", "Note", "If", "In", "An", "And", "Or",
}


class KnowledgeGraphService:
    def __init__(self, config):
        self.config = config
        self.out_dir = Path(config["KG_OUTPUT_DIR"])
        self.chunk_size = config["CHUNK_SIZE"]
        self.chunk_overlap = config["CHUNK_OVERLAP"]
        self.embed_model_name = config["EMBED_MODEL"]
        self.embed_batch = config["EMBED_BATCH"]

    # ── Public entry point ──────────────────────────────────────────────────
    def build(self, source_path: str, kg_name: str = "default",
              upload: bool = False) -> dict:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        self.kg_name = kg_name  # store for _write_manifest / _artifact_key_mapping
        output_dir = self.out_dir / kg_name
        log.info("=== Knowledge Graph build started ===")
        log.info("  source_path : %s", src)
        log.info("  kg_name     : %s", kg_name)
        log.info("  output_dir  : %s", output_dir)
        log.info("  upload      : %s", upload)
        log.info("  embed_model : %s", self.embed_model_name)

        self._ensure_dirs()

        t0 = time.time()
        log.info("Step 1/4: Collecting chunks from source files …")
        chunks = self._collect_chunks(src)
        if not chunks:
            raise ValueError(f"No ingestible text files found under: {source_path}")
        log.info("  → %d chunks collected in %.1fs", len(chunks), time.time() - t0)

        chunks_path = self._save_chunks(chunks)
        log.info("  → chunks saved to %s", chunks_path)

        t1 = time.time()
        log.info("Step 2/4: Building NetworkX knowledge graph …")
        graph, entity_map, stats = self._build_graph(chunks)
        log.info("  → graph built: %d nodes, %d edges in %.1fs",
                 stats["nodes"], stats["edges"], time.time() - t1)

        kg_path, e2c_path = self._save_graph(graph, entity_map)
        log.info("  → graph saved to %s", kg_path)

        t2 = time.time()
        log.info("Step 3/4: Building FAISS index (embedding %d chunks) …",
                 stats["chunks"])
        faiss_path, node_map_path, embed_dim = self._build_faiss(graph)
        log.info("  → FAISS index built: dim=%d in %.1fs",
                 embed_dim, time.time() - t2)

        t3 = time.time()
        log.info("Step 4/4: Writing manifest …")
        manifest_path = self._write_manifest(stats, embed_dim, source_path)
        log.info("  → manifest saved to %s", manifest_path)

        artifacts = {
            "manifest": str(manifest_path),
            "chunks": str(chunks_path),
            "graph": str(kg_path),
            "entity_map": str(e2c_path),
            "faiss_index": str(faiss_path),
            "node_id_map": str(node_map_path),
        }

        uploaded = []
        storage = None
        if upload:
            log.info("Step 5/5: Uploading artifacts to HuggingFace …")
            t_u = time.time()
            uploaded = self._upload(artifacts)
            storage = self._storage_info()
            log.info("  → %d artifact(s) uploaded in %.1fs",
                     len(uploaded), time.time() - t_u)

        elapsed = time.time() - t0
        log.info("=== Knowledge Graph build complete in %.1fs ===", elapsed)
        log.info("  KG name     : %s", kg_name)
        log.info("  Chunks      : %d", stats["chunks"])
        log.info("  Nodes       : %d  (types: %s)",
                 stats["nodes"], stats.get("node_types", {}))
        log.info("  Edges       : %d", stats["edges"])
        log.info("  Embed dim   : %d", embed_dim)
        log.info("  Local dir   : %s", output_dir)

        result = {"stats": stats, "artifacts": artifacts, "uploaded": uploaded}
        if storage is not None:
            result["storage"] = storage
        return result

    # ── Step 1: read + chunk files ──────────────────────────────────────────
    def _collect_chunks(self, src: Path) -> list[dict]:
        files = [
            Path(p) for p in glob.glob(str(src / "**" / "*"), recursive=True)
            if Path(p).is_file() and Path(p).suffix.lower() in TEXT_EXTENSIONS
        ]

        all_chunks: list[dict] = []
        chunk_id = 0
        for fpath in files:
            text = self._read_text(fpath)
            if not text.strip():
                continue
            heading = fpath.stem
            for piece in self._chunk_text(text):
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": len([c for c in all_chunks
                                        if c["source_file"] == str(fpath)]),
                    "text": piece,
                    "heading": heading,
                    "doc_title": fpath.name,
                    "source_file": str(fpath.relative_to(src)),
                    "version": "v1",
                })
                chunk_id += 1
        return all_chunks

    @staticmethod
    def _read_text(fpath: Path) -> str:
        raw = fpath.read_text(encoding="utf-8", errors="ignore")
        if fpath.suffix.lower() in (".html", ".htm"):
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw, "lxml")
                for tag in soup.find_all(["nav", "footer", "script", "style",
                                          "header", "aside"]):
                    tag.decompose()
                return soup.get_text(separator=" ", strip=True)
            except Exception:
                return re.sub(r"<[^>]+>", " ", raw)
        return raw

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        word_size = int(self.chunk_size / 0.75)
        word_overlap = int(self.chunk_overlap / 0.75)
        chunks, start = [], 0
        while start < len(words):
            end = min(start + word_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += word_size - word_overlap
        return chunks

    def _save_chunks(self, chunks: list[dict]) -> Path:
        path = self.out_dir / self.kg_name / "chunks" / "chunks.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        return path

    # ── Step 2: build the knowledge graph ───────────────────────────────────
    def _build_graph(self, chunks: list[dict]):
        import networkx as nx

        extract = self._build_entity_extractor()

        G = nx.DiGraph()
        entity_to_chunks: dict[str, list[int]] = defaultdict(list)
        doc_nodes, section_nodes = {}, {}

        for chunk in chunks:
            cid, src = chunk["chunk_id"], chunk["source_file"]
            heading, title = chunk["heading"], chunk["doc_title"]
            version = chunk["version"]

            doc_key = (version, src)
            if doc_key not in doc_nodes:
                doc_id = f"doc::{version}::{Path(src).stem}"
                G.add_node(doc_id, type="doc", title=title,
                           version=version, source_file=src)
                doc_nodes[doc_key] = doc_id
            doc_id = doc_nodes[doc_key]

            sec_key = (version, src, heading)
            if sec_key not in section_nodes:
                sec_id = f"sec::{version}::{Path(src).stem}::{heading[:60]}"
                G.add_node(sec_id, type="section", heading=heading,
                           version=version, doc_title=title)
                G.add_edge(doc_id, sec_id, rel="has_section")
                section_nodes[sec_key] = sec_id
            sec_id = section_nodes[sec_key]

            chunk_id = f"chunk::{cid}"
            G.add_node(chunk_id, type="chunk", chunk_id=cid, text=chunk["text"],
                       heading=heading, doc_title=title, version=version,
                       source_file=src, chunk_index=chunk["chunk_index"])
            G.add_edge(sec_id, chunk_id, rel="has_chunk")

            entity_list = list(extract(chunk["text"]))
            for ent in entity_list:
                ent_id = f"entity::{ent}"
                if not G.has_node(ent_id):
                    G.add_node(ent_id, type="entity", name=ent)
                G.add_edge(chunk_id, ent_id, rel="contains_entity")
                G.add_edge(ent_id, chunk_id, rel="mentioned_in")
                entity_to_chunks[ent].append(cid)

            for i in range(len(entity_list)):
                for j in range(i + 1, min(i + 5, len(entity_list))):
                    e1 = f"entity::{entity_list[i]}"
                    e2 = f"entity::{entity_list[j]}"
                    if not G.has_edge(e1, e2):
                        G.add_edge(e1, e2, rel="co_occurs_with", weight=1)
                    else:
                        G[e1][e2]["weight"] = G[e1][e2].get("weight", 1) + 1

        node_types: dict[str, int] = {}
        for _, data in G.nodes(data=True):
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        stats = {
            "chunks": len(chunks),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "node_types": node_types,
        }
        return G, dict(entity_to_chunks), stats

    def _build_entity_extractor(self):
        """Return an extract(text) -> set[str] function (spaCy if available)."""
        nlp = None
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser"])
            nlp.max_length = 2_000_000
        except Exception:
            nlp = None  # fall back to regex-only extraction

        def extract(text: str) -> set[str]:
            entities: set[str] = set()
            if nlp is not None:
                doc = nlp(text[:100_000])
                for ent in doc.ents:
                    if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                        entities.add(ent.text.strip())
            for match in ENTITY_PATTERN.finditer(text):
                token = match.group().strip()
                if token and token not in STOP_PASCAL and len(token) > 2:
                    entities.add(token)
            return entities

        return extract

    def _save_graph(self, graph, entity_map):
        kg_path = self.out_dir / self.kg_name / "kg" / "graph.pkl"
        with open(kg_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        e2c_path = self.out_dir / self.kg_name / "kg" / "entity_to_chunks.json"
        with open(e2c_path, "w", encoding="utf-8") as f:
            json.dump(entity_map, f)
        return kg_path, e2c_path

    # ── Step 3: embeddings + FAISS index ────────────────────────────────────
    def _build_faiss(self, graph):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        chunk_nodes = [(nid, d) for nid, d in graph.nodes(data=True)
                       if d.get("type") == "chunk"]
        chunk_nodes.sort(key=lambda x: x[1]["chunk_id"])
        node_ids = [nid for nid, _ in chunk_nodes]
        texts = [f"passage: {d['text']}" for _, d in chunk_nodes]

        embedder = SentenceTransformer(self.embed_model_name)
        embeddings = embedder.encode(
            texts, batch_size=self.embed_batch, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

        dim = int(embeddings.shape[1])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        ids = np.array([d["chunk_id"] for _, d in chunk_nodes], dtype=np.int64)
        index.add_with_ids(embeddings, ids)

        faiss_path = self.out_dir / self.kg_name / "faiss" / "index.faiss"
        faiss.write_index(index, str(faiss_path))
        node_map_path = self.out_dir / self.kg_name / "faiss" / "node_id_map.json"
        with open(node_map_path, "w", encoding="utf-8") as f:
            json.dump(node_ids, f)
        return faiss_path, node_map_path, dim

    # ── Step 4: manifest ────────────────────────────────────────────────────
    def _write_manifest(self, stats, embed_dim, source_path) -> Path:
        manifest = {
            "source_path": str(source_path),
            "total_chunks": stats["chunks"],
            "total_nodes": stats["nodes"],
            "total_edges": stats["edges"],
            "embed_model": self.embed_model_name,
            "embed_dim": embed_dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files": {
                "chunks": "data/repo-kg/chunks/chunks.jsonl",
                "graph": "data/repo-kg/kg/graph.pkl",
                "entity_map": "data/repo-kg/kg/entity_to_chunks.json",
                "faiss_index": "data/repo-kg/faiss/index.faiss",
                "node_id_map": "data/repo-kg/faiss/node_id_map.json",
            },
        }
        path = self.out_dir / self.kg_name / "manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return path

    # ── Step 5: upload to HuggingFace ───────────────────────────────────────
    def _artifact_key_mapping(self) -> dict:
        """Map artifact keys → destination paths (shared by bucket & repo)."""
        data_dir = self.config["HF_DATA_DIR"]
        return {
            "manifest": f"{data_dir}/manifest.json",
            "chunks": f"{data_dir}/chunks/chunks.jsonl",
            "graph": f"{data_dir}/kg/graph.pkl",
            "entity_map": f"{data_dir}/kg/entity_to_chunks.json",
            "faiss_index": f"{data_dir}/faiss/index.faiss",
            "node_id_map": f"{data_dir}/faiss/node_id_map.json",
        }

    def _storage_info(self) -> dict:
        """Describe where artifacts were uploaded (for clients / cleanup)."""
        backend = str(self.config.get("HF_STORAGE_BACKEND", "bucket")).lower()
        if backend == "bucket":
            return {"backend": "bucket", "id": self.config["HF_BUCKET_ID"]}
        return {
            "backend": "repo",
            "id": self.config["HF_REPO_ID"],
            "repo_type": self.config["HF_REPO_TYPE"],
        }

    def _upload(self, artifacts: dict) -> list[str]:
        token = self.config["HF_TOKEN"]
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set — cannot upload artifacts. "
                "Set it in backend/.env or pass upload=false."
            )

        backend = str(self.config.get("HF_STORAGE_BACKEND", "bucket")).lower()
        if backend == "bucket":
            return self._upload_bucket(artifacts, token)
        return self._upload_repo(artifacts, token)

    def _upload_bucket(self, artifacts: dict, token: str) -> list[str]:
        """Upload artifacts to a HuggingFace Storage Bucket."""
        from huggingface_hub import batch_bucket_files, create_bucket

        bucket_id = self.config["HF_BUCKET_ID"]
        mapping = self._artifact_key_mapping()

        # Ensure the bucket exists (no-op if it already does).
        if self.config.get("HF_BUCKET_AUTO_CREATE", True):
            try:
                create_bucket(
                    bucket_id,
                    private=bool(self.config.get("HF_BUCKET_PRIVATE", True)),
                    exist_ok=True,
                    token=token,
                )
            except Exception:
                # Bucket may already exist / lack create perms — keep going and
                # let the upload surface any real error.
                pass

        add = [(artifacts[key], dest) for key, dest in mapping.items()]
        batch_bucket_files(bucket_id, add=add, token=token)
        return [dest for _, dest in mapping.items()]

    def _upload_repo(self, artifacts: dict, token: str) -> list[str]:
        """Commit artifacts into the Space/dataset git repo (Files tab)."""
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        repo_id = self.config["HF_REPO_ID"]
        repo_type = self.config["HF_REPO_TYPE"]
        mapping = self._artifact_key_mapping()

        uploaded = []
        for key, repo_path in mapping.items():
            local_path = artifacts[key]
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"[pipeline] Upload {os.path.basename(local_path)}",
            )
            uploaded.append(repo_path)
        return uploaded

    # ── helpers ─────────────────────────────────────────────────────────────
    def _ensure_dirs(self):
        for sub in ("chunks", "kg", "faiss"):
            (self.out_dir / self.kg_name / sub).mkdir(parents=True, exist_ok=True)
