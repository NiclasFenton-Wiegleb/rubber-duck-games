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
    HF_DATA_DIR = os.getenv("HF_DATA_DIR", "data")  # path-in-repo prefix

    # ── Knowledge Graph pipeline ────────────────────────────────────────────
    # Local working directory where artifacts are written before upload.
    KG_OUTPUT_DIR = Path(os.getenv("KG_OUTPUT_DIR", BASE_DIR / "artifacts"))

    # Default folder to ingest if the request doesn't specify one.
    KG_SOURCE_PATH = os.getenv("KG_SOURCE_PATH", str(BASE_DIR.parent / "test-repo"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))      # target tokens / chunk
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))  # overlap tokens

    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

    # ── Small Language Model ────────────────────────────────────────────────
    # The fine-tuned SmolLM3-3B produced by the notebook pipeline.
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "niclasfw/smollm3-3b-codex")
    LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")  # auto | cpu | cuda
    LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
    LLM_COMPLEX_MAX_NEW_TOKENS = int(os.getenv("LLM_COMPLEX_MAX_NEW_TOKENS", "2048"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    # Lazily load the model on first request rather than at startup.
    LLM_LAZY_LOAD = os.getenv("LLM_LAZY_LOAD", "true").lower() == "true"
