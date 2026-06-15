"""
Central configuration for the Rubber Duck Games backend.

Values are read from environment variables (see .env.example) with sensible
defaults. Most knowledge-graph / model constants mirror the values used in the
original RubberDuckGames.ipynb pipeline so behaviour stays consistent.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from backend/.env if present.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Config:
    # ── Flask server ────────────────────────────────────────────────────────
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    # Frontend origins allowed to call the local APIs.
    CORS_ORIGINS = _split_csv(
        os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
    )

    # ── HuggingFace ─────────────────────────────────────────────────────────
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("HF_REPO_ID", "niclasfw/rubber-duck-games")
    HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "space")  # space | dataset | model
    HF_DATA_DIR = os.getenv("HF_DATA_DIR", "data")  # path-in-repo / bucket-key prefix

    # Where the KG artifacts are stored:
    #   "bucket" → HuggingFace Storage Bucket (huggingface.co/buckets/<id>)
    #   "repo"   → committed into the Space/dataset git repo (Files tab)
    HF_STORAGE_BACKEND = os.getenv("HF_STORAGE_BACKEND", "bucket").lower()
    # Bucket id (namespace/name). Defaults to the same id as the repo.
    HF_BUCKET_ID = os.getenv("HF_BUCKET_ID", HF_REPO_ID)
    # Create the bucket automatically if it doesn't exist yet.
    HF_BUCKET_AUTO_CREATE = os.getenv("HF_BUCKET_AUTO_CREATE", "true").lower() == "true"
    HF_BUCKET_PRIVATE = os.getenv("HF_BUCKET_PRIVATE", "true").lower() == "true"

    # ── Knowledge Graph pipeline ────────────────────────────────────────────
    # Local working directory where artifacts are written before upload.
    KG_OUTPUT_DIR = Path(os.getenv("KG_OUTPUT_DIR", BASE_DIR / "artifacts"))

    # Default folder to ingest if the request doesn't specify one.
    KG_SOURCE_PATH = os.getenv("KG_SOURCE_PATH", str(BASE_DIR.parent / "test-repo"))

    # Name of the knowledge graph being built. Becomes its subfolder name in
    # both the local artifacts dir and the storage bucket so multiple KGs can
    # coexist, e.g. <storage>/godot-docs/faiss/index.faiss.
    KG_NAME = os.getenv("KG_NAME", "default")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))      # target tokens / chunk
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))  # overlap tokens

    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
    # Device used to embed chunks when building a knowledge graph.
    # IMPORTANT: the KG build runs inside the Flask backend thread, which on a
    # HuggingFace ZeroGPU Space is *outside* any @spaces.GPU context. CUDA is
    # only available inside that context, so creating a SentenceTransformer that
    # auto-selects "cuda" here triggers a low-level CUDA init that ZeroGPU's
    # emulation layer rejects. Default to CPU so the build always works; small
    # chunk sets embed quickly on CPU. Override with EMBED_DEVICE=cuda only on a
    # dedicated-GPU Space where CUDA is always attached.
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")


    # ── Retrieval (multi-KG context for queries) ────────────────────────────
    # Root of the attached storage that holds one or more knowledge graphs,
    # each in its own subfolder (<root>/<kg-name>/{faiss,chunks,manifest.json}).
    # On a HuggingFace Space with the bucket mounted this is typically "/data";
    # locally we fall back to KG_OUTPUT_DIR.
    KG_STORAGE_DIR = Path(
        os.getenv("KG_STORAGE_DIR", os.getenv("KG_OUTPUT_DIR", str(BASE_DIR / "artifacts")))
    )
    # How many chunks to pull from each KG before merging.
    RETRIEVAL_PER_KG_K = int(os.getenv("RETRIEVAL_PER_KG_K", "8"))
    # How many chunks to keep in the final merged context across all KGs.
    # Trimmed from 4 → 3: every extra chunk (~400 tokens) inflates prefill and
    # is re-attended on every generated token, so fewer chunks = faster RAG.
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    # Hard cap on characters kept per chunk before it is injected into the
    # prompt. Long chunks get truncated so a single oversized snippet can't blow
    # up the context window. 0 disables the per-chunk cap.
    RETRIEVAL_MAX_CHUNK_CHARS = int(os.getenv("RETRIEVAL_MAX_CHUNK_CHARS", "1200"))
    # Hard cap on the total merged context (across all kept chunks). Keeps the
    # input length — and therefore prefill time — bounded. 0 disables the cap.
    RETRIEVAL_MAX_CONTEXT_CHARS = int(os.getenv("RETRIEVAL_MAX_CONTEXT_CHARS", "3000"))
    # Minimum FAISS similarity score a hit must clear to be injected. Acts as a
    # "free" relevance classifier (reusing the embedder we already load): if the
    # best chunk is weaker than this, we drop context entirely rather than feed
    # the model irrelevant text. Tune per embedding model; 0 disables the gate.
    RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.30"))


    # ── Knowledge-graph augmented retrieval ─────────────────────────────────
    # Each KG ships a NetworkX graph (kg/graph.pkl) + an entity→chunk-ids map
    # (kg/entity_to_chunks.json) alongside its FAISS index. When enabled, the
    # retrieval service loads those into RAM at startup and uses them to (a)
    # expand the FAISS seed hits with entity-linked neighbour chunks the vector
    # search missed and (b) boost chunks that share entities with other strong
    # hits. All graph ops are in-memory dict/edge lookups (no CUDA, no model),
    # so query latency stays effectively flat. Master switch — off falls back to
    # pure vector search (the prior behaviour).
    KG_EXPANSION_ENABLED = os.getenv("KG_EXPANSION_ENABLED", "true").lower() == "true"
    # Max entity-linked neighbour chunks pulled in per FAISS seed hit. Keeps the
    # candidate set (and therefore context size) bounded. 0 disables expansion
    # while still allowing entity re-ranking.
    KG_EXPANSION_MAX_NEIGHBORS = int(os.getenv("KG_EXPANSION_MAX_NEIGHBORS", "5"))
    # Score bonus added per shared entity when re-ranking candidates. Small by
    # design so semantic (FAISS) similarity stays the dominant signal. 0 disables
    # entity re-ranking.
    KG_ENTITY_BOOST = float(os.getenv("KG_ENTITY_BOOST", "0.05"))
    # Cap on the total bonus a single chunk can accrue from shared entities, so a
    # heavily-connected hub chunk can't dominate purely on connectivity.
    KG_ENTITY_BOOST_CAP = float(os.getenv("KG_ENTITY_BOOST_CAP", "0.20"))



    # ── Prompting ───────────────────────────────────────────────────────────
    # Local text file holding the system prompt prepended to every query.
    SYSTEM_PROMPT_PATH = os.getenv(
        "SYSTEM_PROMPT_PATH", str(BASE_DIR / "prompts" / "system_prompt.txt")
    )

    # ── Language Model ──────────────────────────────────────────────────────
    # Any HuggingFace model id compatible with AutoModelForCausalLM.
    # Gemma 4 (and newer models) require AutoProcessor instead of AutoTokenizer.
    # Set via LLM_MODEL_ID env var.
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "niclasfw/smollm3-3b-codex")
    LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")  # auto | cpu | cuda | mps
    # Output length is the single biggest wall-clock lever (every token is a
    # forward pass), so the simple/fast path defaults to a tighter budget. The
    # JSON early-stop (stop_on_json) usually halts well before this anyway.
    LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))

    LLM_COMPLEX_MAX_NEW_TOKENS = int(os.getenv("LLM_COMPLEX_MAX_NEW_TOKENS", "2048"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    # Lazily load the model on first request rather than at startup.
    LLM_LAZY_LOAD = os.getenv("LLM_LAZY_LOAD", "true").lower() == "true"
    # Load the model weights in 4-bit (bitsandbytes NF4). This quarters VRAM and
    # speeds up generation on a single GPU (e.g. the fine-tuned SmolLM3 on a
    # ZeroGPU instance). Requires CUDA + bitsandbytes; on CPU it is silently
    # ignored and the model loads in its normal precision.
    LLM_LOAD_IN_4BIT = os.getenv("LLM_LOAD_IN_4BIT", "true").lower() == "true"


