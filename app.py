"""
HuggingFace Space entry point — Rubber Duck Games.

This single process does three things on startup:

  1. Configures the environment for a GPU Space (mounts the storage bucket path,
     selects CUDA, eager-loads the model) and boots the Flask backend in a
     background thread so its local APIs are available at 127.0.0.1:5000.
  2. Warms up the heavy pieces once: the system prompt (from file), the
     knowledge-graph retrieval indexes (from the mounted storage), and the small
     language model (onto the GPU). Because Flask runs in the *same* process,
     these singletons are shared with the API handlers.
  3. Serves a minimal Gradio UI (a single text box) that submits prompts through
     the "simple request" workstream by calling the local Flask API, and shows
     the raw response for testing / debugging.

HuggingFace runs this file because README.md declares `sdk: gradio` and
`app_file: app.py`. Gradio binds the public port (7860); Flask stays internal.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

# Make the backend package importable (config.py, api/...).
sys.path.insert(0, str(BACKEND_DIR))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("space")


# ── Environment defaults (set BEFORE importing backend config) ───────────────
def _configure_environment() -> str:
    """Pick sensible Space defaults; honour anything already set in the env."""
    os.environ.setdefault("HOST", "127.0.0.1")
    os.environ.setdefault("PORT", "5000")
    # Load the model at startup (not lazily) so the first request is fast.
    os.environ.setdefault("LLM_LAZY_LOAD", "false")
    # Flask debug reloader must stay off inside a background thread.
    os.environ.setdefault("FLASK_DEBUG", "false")

    # Knowledge-graph storage: HF persistent storage is mounted at /data.
    if "KG_STORAGE_DIR" not in os.environ:
        default_storage = "/data" if Path("/data").exists() else str(BACKEND_DIR / "artifacts")
        os.environ["KG_STORAGE_DIR"] = default_storage

    # Select the GPU when available.
    if "LLM_DEVICE" not in os.environ:
        try:
            import torch  # noqa: WPS433 (lazy, optional)
            os.environ["LLM_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            os.environ["LLM_DEVICE"] = "auto"

    return os.environ["KG_STORAGE_DIR"]


STORAGE_DIR = _configure_environment()
PORT = int(os.environ["PORT"])
BACKEND_URL = f"http://127.0.0.1:{PORT}"


# ── Load the Flask backend (backend/app.py) ──────────────────────────────────
def _load_backend():
    """Import backend/app.py explicitly (avoids the app.py name clash)."""
    spec = importlib.util.spec_from_file_location("backend_app", BACKEND_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backend_module = _load_backend()
flask_app = backend_module.create_app()


# ── Shared warm-up state (surfaced in the UI for debugging) ──────────────────
STATE: dict = {"ready": False, "error": None, "kgs": [], "device": os.environ.get("LLM_DEVICE")}


def _run_backend():
    """Serve the Flask API on the internal port (single worker, no reloader)."""
    log.info("Starting Flask backend on %s", BACKEND_URL)
    flask_app.run(
        host=os.environ["HOST"],
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


def _warmup():
    """Mount/verify storage, load the system prompt, KG indexes and the model."""
    cfg = flask_app.config
    try:
        from api.services.prompt_service import get_system_prompt
        from api.services.retrieval_service import get_retrieval_service
        from api.services.llm_service import get_llm_service

        log.info("Warm-up: loading system prompt …")
        prompt = get_system_prompt(cfg)
        log.info("Warm-up: system prompt loaded (%d chars)", len(prompt))

        log.info("Warm-up: scanning storage for knowledge graphs at %s …", STORAGE_DIR)
        retrieval = get_retrieval_service(cfg)
        retrieval.refresh()
        STATE["kgs"] = retrieval.kg_names
        log.info("Warm-up: knowledge graphs found: %s", STATE["kgs"] or "(none)")

        log.info("Warm-up: loading model onto %s …", STATE["device"])
        get_llm_service(cfg).load()
        log.info("Warm-up: model loaded.")

        STATE["ready"] = True
    except Exception as exc:  # noqa: BLE001 - surface to the UI
        STATE["error"] = str(exc)
        log.exception("Warm-up failed")


def _wait_for_backend(timeout: float = 30.0):
    """Block until the Flask /health endpoint responds (best effort)."""
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{BACKEND_URL}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# Kick off backend + warm-up in the background so the UI can bind quickly.
threading.Thread(target=_run_backend, daemon=True).start()
threading.Thread(target=_warmup, daemon=True).start()


# ── Gradio UI ────────────────────────────────────────────────────────────────
import gradio as gr  # noqa: E402  (after env + backend setup)
import requests  # noqa: E402


def backend_status() -> str:
    parts = []
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        parts.append(f"Backend: {health.get('status', '?')} ✅")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"Backend: down ⏳ ({exc.__class__.__name__})")

    if STATE["ready"]:
        parts.append("Model: loaded ✅")
    elif STATE["error"]:
        parts.append(f"Model: error ❌ {STATE['error']}")
    else:
        parts.append("Model: loading… ⏳")

    parts.append(f"Device: {STATE['device']}")
    parts.append(f"KGs: {', '.join(STATE['kgs']) if STATE['kgs'] else '(none)'}")
    return "  |  ".join(parts)


def ask(message: str, use_context: bool, max_new_tokens: int, temperature: float):
    """Submit a prompt through the simple-request workstream (via the API)."""
    message = (message or "").strip()
    if not message:
        return "Please enter a prompt.", "{}"

    payload = {
        "query": message,
        "use_context": bool(use_context),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }

    try:
        resp = requests.post(f"{BACKEND_URL}/api/query", json=payload, timeout=600)
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Could not reach backend: {exc}", json.dumps({"error": str(exc)}, indent=2)

    raw = json.dumps(data, indent=2, ensure_ascii=False)
    if data.get("status") != "ok":
        return f"⚠️ Backend error: {data.get('message', 'unknown error')}", raw

    answer = data.get("answer", "")
    sources = data.get("sources") or []
    if sources:
        src_lines = "\n".join(
            f"- `{s.get('kg')}` · chunk {s.get('chunk_id')} · score {s.get('score')}"
            for s in sources
        )
        answer = f"{answer}\n\n---\n**Context sources ({len(sources)}):**\n{src_lines}"
    return answer, raw


with gr.Blocks(title="Rubber Duck Games") as demo:
    gr.Markdown("# 🦆 Rubber Duck Games — test console")
    gr.Markdown(
        "Enter a prompt and submit it through the **simple request** workstream "
        "(file system prompt → multi-KG context → local SLM)."
    )

    with gr.Row():
        status = gr.Markdown(value="Status: loading…")
        refresh = gr.Button("↻ Refresh status", scale=0)

    query = gr.Textbox(
        label="Your prompt",
        placeholder="e.g. How do I move a CharacterBody2D in Godot?",
        lines=4,
    )

    with gr.Accordion("Options", open=False):
        use_context = gr.Checkbox(value=True, label="Use knowledge-graph context")
        max_new_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max new tokens")
        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")

    submit = gr.Button("Submit", variant="primary")
    answer = gr.Markdown(label="Answer")

    with gr.Accordion("Raw response (debug)", open=False):
        raw_json = gr.Code(label="JSON", language="json")

    submit.click(
        ask,
        inputs=[query, use_context, max_new_tokens, temperature],
        outputs=[answer, raw_json],
    )
    query.submit(
        ask,
        inputs=[query, use_context, max_new_tokens, temperature],
        outputs=[answer, raw_json],
    )
    refresh.click(lambda: backend_status(), outputs=status)

    # Populate status once on load and poll periodically.
    demo.load(lambda: backend_status(), outputs=status)


if __name__ == "__main__":
    # Give the backend a moment to come up so the first status read is accurate.
    _wait_for_backend(timeout=15)
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
