"""
HuggingFace Space entry point — Rubber Duck Games.

This single process does three things on startup:

  1. Configures the environment for a ZeroGPU Space (mounts the storage bucket
     path, pins CUDA as the model device) and boots the Flask backend in a
     background thread so its local APIs are available at 127.0.0.1:5000.
  2. Warms up the lightweight pieces once: the system prompt (from file) and the
     knowledge-graph retrieval indexes (from the mounted storage). The model is
     NOT loaded here — on ZeroGPU there is no persistent GPU, so the model is
     loaded and run inside a @spaces.GPU function on the first query instead.
  3. Serves a minimal Gradio UI (a single text box) that submits prompts through
     the "simple request" workstream. Generation runs directly inside the
     @spaces.GPU function (`_generate_simple`) so the on-demand GPU is attached
     to the call; the raw response is shown for testing / debugging.

HuggingFace runs this file because README.md declares `sdk: gradio` and
`app_file: app.py`. Gradio binds the public port (7860); Flask stays internal.
"""

from __future__ import annotations

# Import `spaces` FIRST — before torch is imported anywhere — so the ZeroGPU
# runtime can patch torch's CUDA layer. On ZeroGPU there is no persistent GPU;
# one is attached only while a @spaces.GPU-decorated function is executing.
import spaces  # noqa: E402

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
    # ZeroGPU: there is NO persistent GPU at startup, so the model must not be
    # eager-loaded (that would initialise CUDA in the parent process and break
    # ZeroGPU's on-demand allocation). Force lazy load — the weights are loaded
    # on the first @spaces.GPU call instead.
    os.environ["LLM_LAZY_LOAD"] = "true"
    # Flask debug reloader must stay off inside a background thread.
    os.environ.setdefault("FLASK_DEBUG", "false")

    # Knowledge-graph storage: HF persistent storage is mounted at /data.
    if "KG_STORAGE_DIR" not in os.environ:
        default_storage = "/data" if Path("/data").exists() else str(BACKEND_DIR / "artifacts")
        os.environ["KG_STORAGE_DIR"] = default_storage

    # ZeroGPU: a CUDA device is only visible *inside* a @spaces.GPU function, so
    # torch.cuda.is_available() is False at startup. Pin the device to "cuda"
    # unconditionally — the model is loaded and run within the GPU-decorated
    # path (_generate_simple), where the device is actually attached. We avoid
    # importing torch / probing CUDA here to keep the parent process CUDA-clean.
    os.environ.setdefault("LLM_DEVICE", "cuda")

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
    """Mount/verify storage, load the system prompt and the KG indexes.

    NOTE: We deliberately do NOT load the model here. On ZeroGPU the model must
    be loaded and run inside a @spaces.GPU function (`_generate_simple`) so that
    a GPU is actually attached; loading it now (in this background thread) would
    initialise CUDA in the parent process and break ZeroGPU's on-demand
    allocation. The model therefore loads lazily on the first query.
    """
    cfg = flask_app.config
    try:
        from api.services.prompt_service import get_system_prompt
        from api.services.retrieval_service import get_retrieval_service

        log.info("Warm-up: loading system prompt …")
        prompt = get_system_prompt(cfg)
        log.info("Warm-up: system prompt loaded (%d chars)", len(prompt))

        log.info("Warm-up: scanning storage for knowledge graphs at %s …", STORAGE_DIR)
        retrieval = get_retrieval_service(cfg)
        retrieval.refresh()
        STATE["kgs"] = retrieval.kg_names
        log.info("Warm-up: knowledge graphs found: %s", STATE["kgs"] or "(none)")

        log.info("Warm-up: model will load lazily on the first GPU request.")
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


# ── ZeroGPU duration / quota ─────────────────────────────────────────────────
# `duration` is NOT "how long the GPU runs" — it is the amount of quota ZeroGPU
# RESERVES UP FRONT for the call. Any unused portion is refunded when the call
# returns, BUT if `duration` exceeds the quota you have left *right now*, the
# call is rejected outright with:
#     "exceeded your ZeroGPU quota (<duration>s requested vs. <remaining>s left)"
# So we keep this modest (and tunable) instead of reserving a big block. The
# first call still has to load the multi-GB model, so it needs more headroom
# than later calls — hence a separate, larger first-load reservation.
GPU_DURATION = int(os.getenv("ZEROGPU_DURATION", "60"))
GPU_LOAD_DURATION = int(os.getenv("ZEROGPU_LOAD_DURATION", "120"))


def _gpu_duration(message: str, use_context: bool, max_new_tokens: int,
                  temperature: float) -> int:
    """Reserve more quota only for the (one-off) first call that loads weights."""
    try:
        from api.services.llm_service import get_llm_service

        if not get_llm_service(flask_app.config).is_loaded:
            return GPU_LOAD_DURATION
    except Exception:
        return GPU_LOAD_DURATION
    return GPU_DURATION


@spaces.GPU(duration=_gpu_duration)
def _generate_simple(message: str, use_context: bool, max_new_tokens: int,
                     temperature: float) -> dict:
    """Run one "simple request" generation on an on-demand ZeroGPU device.

    This is the ONLY place the model touches CUDA. ZeroGPU attaches a GPU for
    the duration of this call, so BOTH the first-time model load and every
    generation must happen inside it — not in the background Flask thread, whose
    separate call stack would NOT have a GPU attached. We therefore call the
    in-process LLM service singleton directly here rather than going through the
    internal Flask HTTP endpoint.

    The reserved quota is computed by `_gpu_duration`: a larger block for the
    first call (which downloads/loads the multi-GB model) and a small block for
    every subsequent generation, so we don't needlessly exhaust the quota.
    """
    from api.services.llm_service import get_llm_service


    llm = get_llm_service(flask_app.config)
    return llm.generate_simple(
        message,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        use_context=bool(use_context),
    )


def ask(message: str, use_context: bool, max_new_tokens: int, temperature: float):
    """Submit a prompt through the simple-request workstream (on ZeroGPU)."""
    message = (message or "").strip()
    if not message:
        return "Please enter a prompt.", "{}"

    try:
        result = _generate_simple(message, use_context, max_new_tokens, temperature)
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Generation failed: {exc}", json.dumps({"error": str(exc)}, indent=2)

    data = {"status": "ok", "mode": "simple", "query": message, **result}
    raw = json.dumps(data, indent=2, ensure_ascii=False)

    answer = result.get("answer", "")
    sources = result.get("sources") or []
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
