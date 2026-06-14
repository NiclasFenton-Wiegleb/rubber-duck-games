"""
HuggingFace Space entry point — Rubber Duck Games.

This single process does three things on startup:

  1. Configures the environment for a CPU-only Space (mounts the storage bucket
     path, pins CPU as the model device) and boots the Flask backend in a
     background thread so its local APIs are available at 127.0.0.1:5000.
  2. Warms up everything once: the system prompt (from file), the knowledge-graph
     retrieval indexes (from the mounted storage) and the model itself. On this
     CPU-only runtime the model is eager-loaded at launch so the weights are
     resident in memory before the first user query.
  3. Serves a minimal Gradio UI (a single text box) that submits prompts through
     the "simple request" workstream; the raw response is shown for testing /
     debugging.


HuggingFace runs this file because README.md declares `sdk: gradio` and
`app_file: app.py`. Gradio binds the public port (7860); Flask stays internal.
"""

from __future__ import annotations

import importlib.util
import html
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    force=True,
)
log = logging.getLogger("space")

# ── Optional HuggingFace ZeroGPU import ──────────────────────────────────────
# On HuggingFace ZeroGPU, `spaces` must be imported FIRST — before torch — so
# the runtime can patch torch's CUDA layer. When running locally, `spaces` is
# not installed and we run model inference on the CPU (or local GPU) directly.
try:
    import spaces  # noqa: E402

    _SPACES_AVAILABLE = True
    log.info("spaces module imported — running on HuggingFace ZeroGPU")
except ImportError:
    _SPACES_AVAILABLE = False
    log.info("spaces module not found — running in local mode (no ZeroGPU)")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

# Make the backend package importable (config.py, api/...).
sys.path.insert(0, str(BACKEND_DIR))

from api.services.structured_output import (  # noqa: E402
    normalize_structured_answer,
    parse_partial_json,
    render_duck_questions,
    render_fix_options,
    render_refactor,
    render_repo_findings,
)



# ── Environment defaults (set BEFORE importing backend config) ───────────────
def _configure_environment() -> str:
    """Pick sensible Space defaults; honour anything already set in the env."""
    os.environ.setdefault("HOST", "127.0.0.1")
    os.environ.setdefault("PORT", "5000")
    # Flask debug reloader must stay off inside a background thread.
    os.environ.setdefault("FLASK_DEBUG", "false")

    # Knowledge-graph storage: HF persistent storage is mounted at /data.
    if "KG_STORAGE_DIR" not in os.environ:
        default_storage = "/data" if Path("/data").exists() else str(BACKEND_DIR / "artifacts")
        os.environ["KG_STORAGE_DIR"] = default_storage

    # ── GPU detection at startup ──────────────────────────────────────────
    # Probe the hardware and set LLM_DEVICE accordingly, so the backend loads
    # the model onto the best available accelerator right from the start.
    #
<<<<<<< HEAD
    # ZeroGPU note: on a ZeroGPU Space the `spaces` runtime enables a torch
    # "CUDA emulation" mode in the main process. The *recommended* pattern is to
    # place the model on `cuda` at startup (module level) — the emulation packs
    # the weights so each forked @spaces.GPU worker can restore them quickly.
    # We therefore declare the device as "cuda" on ZeroGPU, but we must NOT call
    # `torch.cuda.get_device_properties()` / other manual probes here: those
    # force a *real* CUDA init in the main process, which is unnecessary (the
    # emulation already handles the load) and can interfere with the worker.
    if "LLM_DEVICE" not in os.environ:
        if _SPACES_AVAILABLE:
            log.info("ZeroGPU runtime detected — setting LLM_DEVICE=cuda; the "
                     "model is eager-loaded at startup and packed by the spaces "
                     "CUDA-emulation layer (no manual torch.cuda probing).")
            os.environ["LLM_DEVICE"] = "cuda"
        else:
            try:
                import torch  # noqa: F811 (already imported above via spaces?)
=======
    # ‼️ IMPORTANT: On HuggingFace ZeroGPU the ``spaces`` module patches
    # ``torch.cuda.is_available()`` to return True in the host process, but
    # CUDA is only real inside a ``@spaces.GPU``-decorated worker. Probing
    # CUDA here and acting on the result (eager-loading the model) creates a
    # CUDA context in the host that poisons ZeroGPU's fork-based workers,
    # causing every generation request to fail with:
    #     RuntimeError: No CUDA GPUs are available
    # So we detect ZeroGPU FIRST and skip all torch probing.
    if "LLM_DEVICE" not in os.environ:
        is_zero_gpu = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true", "yes")
        if is_zero_gpu:
            # ZeroGPU: CUDA is real only inside @spaces.GPU workers. Set
            # "cuda" so _resolve_dtype picks bf16 for the model weights, but
            # do NOT probe torch.cuda here — the patched probe lies and
            # acting on it breaks the fork worker.
            log.info("ZeroGPU runtime detected — setting LLM_DEVICE=cuda; "
                     "the model is lazy-loaded inside the GPU context where "
                     "CUDA is real (host-process probing skipped).")
            os.environ["LLM_DEVICE"] = "cuda"
        else:
            try:
                import torch  # noqa: F811
>>>>>>> backend
            except ImportError:
                os.environ["LLM_DEVICE"] = "cpu"
            else:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    log.info("GPU DETECTED: %s (%.1f GiB VRAM) — setting LLM_DEVICE=cuda",
                             props.name, props.total_memory / (1024 ** 3))
                    os.environ["LLM_DEVICE"] = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    log.info("GPU DETECTED: Apple Metal (MPS) — setting LLM_DEVICE=mps")
                    os.environ["LLM_DEVICE"] = "mps"
                else:
                    log.info("No GPU detected — model will run on CPU.")
                    os.environ["LLM_DEVICE"] = "cpu"

    # ── Lazy-load decision ───────────────────────────────────────────────
<<<<<<< HEAD
    # Eager-load the model at startup whenever a GPU is the target device:
    #   * Dedicated GPU instances (T4, A10G, L4): the GPU is always attached,
    #     so eager-loading means the weights are resident before the first query.
    #   * ZeroGPU: HuggingFace explicitly recommends placing the model on `cuda`
    #     at startup so the emulation layer can pack the weights ahead of time.
    #     This keeps the model "warm" — the first @spaces.GPU request restores
    #     packed tensors quickly instead of loading the multi-GB model on demand
    #     (lazy-loading inside @spaces.GPU is discouraged and far less efficient).
    # Only fall back to lazy-loading on CPU/MPS-only runtimes.
    if "LLM_LAZY_LOAD" not in os.environ:
        gpu_target = _SPACES_AVAILABLE or os.environ.get("LLM_DEVICE") == "cuda"
        os.environ["LLM_LAZY_LOAD"] = "false" if gpu_target else "true"


=======
    # On dedicated GPU instances (T4, A10G, etc.) the GPU is always attached,
    # so we can eager-load the model at startup. On ZeroGPU, CUDA is only real
    # inside @spaces.GPU workers, so we ALWAYS lazy-load — loading in the host
    # process creates a CUDA context that poisons the fork-based worker,
    # causing ``RuntimeError: No CUDA GPUs are available`` on every query.
    if "LLM_LAZY_LOAD" not in os.environ:
        is_zero_gpu = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true", "yes")
        if is_zero_gpu:
            os.environ["LLM_LAZY_LOAD"] = "true"
            log.info("ZeroGPU: lazy-load forced to prevent host CUDA "
                     "context from poisoning fork workers.")
        else:
            gpu_always_available = os.environ.get("LLM_DEVICE") == "cuda"
            os.environ["LLM_LAZY_LOAD"] = "false" if gpu_always_available else "true"
>>>>>>> backend

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
    """Mount/verify storage, load the system prompt, the KG indexes and the model.

    The model is eager-loaded at launch (``LLM_LAZY_LOAD`` is false) on every
    GPU-targeted runtime:

      * Dedicated GPU instances (T4, A10G, L4): the GPU is always attached, so
        eager-loading leaves the weights resident before the first user query.
      * ZeroGPU: HuggingFace recommends placing the model on ``cuda`` at startup
        so the spaces CUDA-emulation layer can pack the weights ahead of time —
        each forked @spaces.GPU worker then restores them quickly, keeping the
        first request fast. (The model load happens in the main process, which
        is exactly where the emulation can capture/pack it.)

    Only CPU/MPS-only runtimes fall back to lazy-loading.
    """

    cfg = flask_app.config
    lazy_load = cfg.get("LLM_LAZY_LOAD", True)
    try:
        from api.services.llm_service import get_llm_service
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

        if not lazy_load:
            log.info("Warm-up: loading model into memory …")
            llm = get_llm_service(cfg)
            llm.load()
            if llm.device:
                STATE["device"] = llm.device
            log.info("Warm-up: model loaded on device '%s'.", llm.device)
        else:
            log.info("Warm-up: skipping model load (lazy-load mode — "
                     "model will load on first query inside GPU context)")

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


APP_CSS = """
:root {
    --duck-yellow: #ffd34d;
    --ink: #182230;
    --muted: #607089;
    --line: #d9e1ea;
    --panel: #ffffff;
    --wash: #eef5f1;
    --teal: #45b8ad;
    --stack-gap: 16px;
}

.gradio-container {
    background: linear-gradient(120deg, #f8f5e9 0%, #eaf3f4 52%, #f5f0df 100%);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.app-shell {
    max-width: 1440px;
    margin: 0 auto;
    padding: 18px;
}

.app-grid {
    display: grid;
    grid-template-columns: minmax(250px, 300px) minmax(520px, 1fr);
    gap: 18px;
    align-items: stretch;
}

.side-panel,
.session-panel {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: 0 18px 42px rgba(64, 85, 105, 0.08);
    overflow: hidden;
}

.side-panel {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 16px;
}

.session-panel {
    display: flex;
    flex-direction: column;
    min-height: 760px;
}

.brand-row {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 6px;
}

.duck-mark {
    width: 28px;
    height: 40px;
    border-radius: 50% 50% 45% 45%;
    background: var(--duck-yellow);
    position: relative;
    box-shadow: inset -7px -3px 0 rgba(223, 164, 24, 0.22);
    flex: 0 0 auto;
}

.duck-mark:before {
    content: "";
    position: absolute;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #172033;
    top: 12px;
    right: 8px;
}

.duck-mark:after {
    content: "";
    position: absolute;
    width: 15px;
    height: 7px;
    border-radius: 2px 9px 9px 2px;
    background: #ff8a3d;
    top: 20px;
    right: -11px;
}

.brand-title {
    color: #172033 !important;
    font-size: 20px;
    font-weight: 800;
    line-height: 1.05;
}

.brand-subtitle {
    color: #40516a !important;
    font-size: 12px;
    line-height: 1.3;
    margin-top: 4px;
}

.section-label {
    color: #61718a;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.06em;
    margin: 0 0 2px;
    text-transform: uppercase;
}

.helper-status,
.helper-status p {
    color: #40516a !important;
    font-size: 12px;
    font-weight: 700;
    margin: 6px 0 10px;
}

.backend-status,
.backend-status p {
    color: #243247 !important;
    font-size: 13px;
    font-weight: 750;
    margin: 0 !important;
    opacity: 1 !important;
    padding: 0 !important;
}

.backend-status > div,
.backend-status .prose {
    margin: 0 !important;
    padding: 0 !important;
}

.status-icons {
    display: grid;
    gap: 8px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin: 4px 0 0;
}

.status-chip {
    align-items: center;
    background: #f7fafc;
    border: 1px solid var(--line);
    border-radius: 6px;
    color: #40516a;
    column-gap: 7px;
    display: inline-flex;
    height: 34px;
    justify-content: flex-start;
    min-width: 0;
    padding: 0 10px;
}

.status-chip svg {
    height: 16px;
    flex: 0 0 auto;
    opacity: 1;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 2;
    width: 16px;
}

.status-label {
    color: #243247;
    font-size: 11px;
    font-weight: 800;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.status-chip.ok {
    background: #e8f7f4;
    border-color: #bde8df;
    color: #0b5f58;
}

.status-chip.loading {
    background: #fff6da;
    border-color: #efd68b;
    color: #80530a;
}

.status-chip.error {
    background: #ffe8e8;
    border-color: #f4b8b8;
    color: #9f1f1f;
}

.status-chip.ok svg {
    stroke: #0b5f58;
}

.status-chip.loading svg {
    stroke: #80530a;
}

.status-chip.error svg {
    stroke: #9f1f1f;
}

.session-panel [role="tab"],
.session-panel button[role="tab"],
.session-panel .tab-nav button,
.session-panel .tabs button {
    color: #40516a !important;
    opacity: 1 !important;
}

.session-panel [role="tab"][aria-selected="true"],
.session-panel button[role="tab"][aria-selected="true"],
.session-panel .tab-nav button.selected {
    color: #f15a24 !important;
    font-weight: 800 !important;
}

.session-header {
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    gap: 14px;
    padding: 14px 16px;
}

.status-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 4px rgba(69, 184, 173, 0.14);
    display: inline-block;
    margin-right: 9px;
}

.session-title {
    color: #172033 !important;
    font-weight: 800;
    font-size: 16px;
}

.session-subtitle {
    color: #40516a !important;
    font-size: 12px;
    margin-left: 24px;
}

.conversation {
    display: flex;
    flex-direction: column;
    gap: var(--stack-gap) !important;
    padding: 0 16px 16px;
}

.conversation > .block,
.conversation > .form,
.conversation .tabs,
.conversation .accordion {
    margin: 0 !important;
}

.conversation [role="tabpanel"],
.conversation .tabitem {
    background: transparent !important;
    border: 0 !important;
}

.conversation > .block:first-child {
    margin-left: -16px !important;
    margin-right: -16px !important;
    width: calc(100% + 32px) !important;
}

.stack-list,
.chat-thread {
    display: flex;
    flex-direction: column;
    gap: var(--stack-gap);
}

.chat-line {
    display: grid;
    grid-template-columns: 34px 1fr;
    gap: 12px;
    margin-bottom: 0;
}

.avatar {
    align-items: center;
    background: var(--duck-yellow);
    border-radius: 50%;
    display: flex;
    font-size: 11px;
    font-weight: 800;
    height: 34px;
    justify-content: center;
    width: 34px;
}

.chat-card,
.info-card {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    color: #172033 !important;
    padding: 12px;
}

.card-kicker {
    color: #40516a !important;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 7px;
}

.chat-card ol,
.chat-card p,
.chat-card li,
.info-card p,
.info-card strong,
.info-card ul,
.info-card ol,
.info-card li {
    color: #243247 !important;
}

.info-card {
    line-height: 1.45;
}

.info-card ul,
.info-card ol {
    margin: 8px 0 10px 20px;
    padding: 0;
}

.info-card li {
    margin: 4px 0;
}

.info-card li,
.info-card li *,
.info-card li::marker {
    color: #243247 !important;
    opacity: 1 !important;
}

.info-card .evidence-list {
    list-style-position: outside;
}

.info-card .evidence-file {
    color: #172033 !important;
    font-weight: 800;
}

.info-card .evidence-symbol,
.info-card .evidence-reason {
    color: #40516a !important;
    font-weight: 650;
}

.info-card .fix-steps,
.info-card .fix-step,
.info-card .fix-step-index,
.info-card .fix-tradeoffs,
.info-card .fix-tradeoff,
.info-card .fix-tradeoffs-heading {
    color: #172033 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #172033 !important;
}

.info-card .fix-steps {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 10px 0 0;
}

.info-card .fix-step {
    display: block;
    line-height: 1.45;
}

.info-card .fix-step-index {
    color: #40516a !important;
    -webkit-text-fill-color: #40516a !important;
    font-weight: 800;
    margin-right: 6px;
}

.info-card .fix-tradeoffs-heading {
    margin: 12px 0 6px;
}

.info-card .fix-tradeoffs {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin: 0 0 4px;
}

.info-card .fix-tradeoff {
    display: block;
    line-height: 1.45;
    padding-left: 14px;
    position: relative;
}

.info-card .fix-tradeoff:before {
    color: #40516a !important;
    content: "•";
    left: 0;
    position: absolute;
}

.prose .info-card .fix-steps,
.prose .info-card .fix-step,
.prose .info-card .fix-tradeoffs,
.prose .info-card .fix-tradeoff,
.prose .info-card .fix-tradeoffs-heading,
.prose .info-card .fix-step-index {
    color: #172033 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #172033 !important;
}

.structured-panel.prose :where(ol, ul, li) {
    color: inherit !important;
    opacity: 1 !important;
}

.repo-pill,
.tag {
    border-radius: 999px;
    display: inline-block;
    font-size: 11px;
    font-weight: 800;
    margin: 4px 5px 0 0;
    padding: 4px 9px;
}

.repo-pill {
    background: #e8f7f4;
    color: #0b5f58;
}

.tag-blue {
    background: #eaf1ff;
    color: #244f9f;
}

.tag-gold {
    background: #fff0cd;
    color: #764812;
}

.tag-green {
    background: #e5f7ef;
    color: #17633e;
}

.confidence-bar {
    background: #edf2f7;
    border-radius: 999px;
    height: 7px;
    margin: 10px 0 8px;
    overflow: hidden;
}

.confidence-bar div {
    background: linear-gradient(90deg, #45b8ad, #5f7df1);
    height: 100%;
    width: 72%;
}

.side-panel,
.session-panel,
.side-panel *,
.session-panel * {
    text-shadow: none;
}

.primary-duck button {
    background: var(--duck-yellow) !important;
    border: 0 !important;
    color: #111827 !important;
    font-weight: 800 !important;
}

.compact-input textarea,
.compact-input input {
    font-size: 13px !important;
}

.hide-label label {
    display: none !important;
}

@media (max-width: 1100px) {
    .app-grid {
        grid-template-columns: 1fr;
    }

    .status-icons {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }

    .session-panel {
        min-height: auto;
    }

}
"""


def backend_status() -> str:
    def icon(name: str) -> str:
        icons = {
            "server": (
                '<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="4" width="18" height="6" rx="2"/>'
                '<rect x="3" y="14" width="18" height="6" rx="2"/><path d="M7 8h.01M7 18h.01"/></svg>'
            ),
            "cpu": (
                '<svg viewBox="0 0 24 24" fill="none"><rect x="7" y="7" width="10" height="10" rx="2"/>'
                '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 15h3M1 9h3M1 15h3"/></svg>'
            ),
            "device": (
                '<svg viewBox="0 0 24 24" fill="none"><rect x="5" y="3" width="14" height="18" rx="2"/>'
                '<path d="M10 18h4"/></svg>'
            ),
            "database": (
                '<svg viewBox="0 0 24 24" fill="none"><ellipse cx="12" cy="5" rx="8" ry="3"/>'
                '<path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
                '<path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>'
            ),
        }
        return icons[name]

    def chip(name: str, state: str, label: str, visible_label: str) -> str:
        safe_label = html.escape(label, quote=True)
        safe_visible_label = html.escape(visible_label)
        return (
            f'<span class="status-chip {state}" title="{safe_label}" aria-label="{safe_label}" '
            f'role="img">{icon(name)}<span class="status-label">{safe_visible_label}</span></span>'
        )

    chips = []
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        backend_state = "ok" if health.get("status") == "ok" else "loading"
        chips.append(chip("server", backend_state, f"Backend: {health.get('status', '?')}", "Backend"))
    except Exception as exc:  # noqa: BLE001
        chips.append(chip("server", "error", f"Backend: down ({exc.__class__.__name__})", "Backend"))

    if STATE["ready"]:
        chips.append(chip("cpu", "ok", "Model: loaded", "Model"))
    elif STATE["error"]:
        chips.append(chip("cpu", "error", f"Model: error {STATE['error']}", "Model"))
    else:
        chips.append(chip("cpu", "loading", "Model: loading", "Model"))

    chips.append(
        chip("device", "ok" if STATE["device"] else "loading", f"Device: {STATE['device'] or 'unknown'}", "Device")
    )
    kg_label = f"Knowledge graphs: {', '.join(STATE['kgs'])}" if STATE["kgs"] else "Knowledge graphs: none"
    chips.append(chip("database", "ok" if STATE["kgs"] else "loading", kg_label, "KGs"))
    return f'<div class="status-icons">{"".join(chips)}</div>'


# ── Model inference ──────────────────────────────────────────────────────────
# Two code paths for running the LLM:
#
#  **ZeroGPU** (HuggingFace Space): wrapped with @spaces.GPU so the ZeroGPU
#    scheduler attaches a GPU for the duration of the call. The first call
#    reserves a larger quota block (model load), subsequent calls use a small
#    reservation so quota isn't wasted.
#
#  **Local** (CPU / local GPU): calls the LLM service directly with no decorator.
#    This is the path taken when the `spaces` module is not installed.
#
# The reserved quota is computed by `_gpu_duration`: a larger block for the
# first call (which downloads/loads the multi-GB model) and a small block for
# every subsequent generation, so we don't needlessly exhaust the quota.
GPU_DURATION = int(os.getenv("ZEROGPU_DURATION", "60"))
GPU_LOAD_DURATION = int(os.getenv("ZEROGPU_LOAD_DURATION", "120"))


def _gpu_duration(message: str, use_context: bool, max_new_tokens: int,
                  temperature: float, stop_on_json: bool = False) -> int:
    """Reserve more quota only for the (one-off) first call that loads weights."""
    try:
        from api.services.llm_service import get_llm_service

        if not get_llm_service(flask_app.config).is_loaded:
            return GPU_LOAD_DURATION
    except Exception:
        return GPU_LOAD_DURATION
    return GPU_DURATION


def _run_generation(message: str, use_context: bool, max_new_tokens: int,
                    temperature: float, stop_on_json: bool = False) -> dict:
    """Core generation logic — called directly or wrapped by @spaces.GPU."""
    from api.services.llm_service import get_llm_service

    log.info("Generation starting: message_len=%d, use_context=%s, "
             "max_new_tokens=%d, temperature=%.2f, stop_on_json=%s",
             len(message), use_context, max_new_tokens, temperature, stop_on_json)

    t0 = time.time()
    llm = get_llm_service(flask_app.config)
    result = llm.generate_simple(
        message,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        use_context=bool(use_context),
        stop_on_json=bool(stop_on_json),
    )
    elapsed = time.time() - t0


    # The device is only known after the (lazy) load runs. Reflect the device
    # the model actually landed on (e.g. "cpu" when no GPU was attached) so the
    # UI status chip is accurate.
    if llm.device:
        STATE["device"] = llm.device

    answer_len = len(result.get("answer", ""))
    sources_count = len(result.get("sources") or [])
    log.info("Generation complete: elapsed=%.1fs, device=%s, answer_len=%d, "
             "sources=%d, context_used=%s",
             elapsed, llm.device, answer_len, sources_count,
             result.get("context_used"))

    return result


# ── Pick the right entry point depending on runtime ──────────────────────────
if _SPACES_AVAILABLE:
    _generate_simple = spaces.GPU(duration=_gpu_duration)(_run_generation)
else:
    _generate_simple = _run_generation


def ask(message: str, use_context: bool, max_new_tokens: int, temperature: float,
        stop_on_json: bool = False):
    """Submit a prompt through the simple-request workstream.

    ``stop_on_json=True`` lets the model stop the moment it has emitted a
    complete JSON object — used by the structured duck-session flows so each
    section returns quickly instead of padding out to the token budget.
    """
    message = (message or "").strip()
    if not message:
        log.warning("ask() received empty message")
        return "Please enter a prompt.", "{}"

    log.info("ask() called: message_len=%d, use_context=%s, max_new=%d, temp=%.2f, "
             "stop_on_json=%s", len(message), use_context, max_new_tokens,
             temperature, stop_on_json)

    try:
        result = _generate_simple(message, use_context, max_new_tokens, temperature,
                                  stop_on_json)
    except Exception as exc:  # noqa: BLE001
        log.exception("Generation failed: %s", exc)
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


def _resolve_destination(destination: str) -> Path:
    destination = (destination or "runs/game-project").strip()
    path = Path(destination).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _run_command(command: list[str], cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    return completed.returncode, output.strip()


def _local_repo_path(repo: str) -> Path:
    path = Path(repo).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def fetch_branches(repo: str):
    repo = (repo or "").strip()
    if not repo:
        return gr.update(choices=["main"], value="main")

    local_path = _local_repo_path(repo)
    if local_path.exists():
        command = ["git", "-C", str(local_path), "branch", "--format", "%(refname:short)"]
    else:
        command = ["git", "ls-remote", "--heads", repo]

    try:
        code, output = _run_command(command, timeout=45)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch branches: %s", exc)
        return gr.update(choices=["main"], value="main")

    if code != 0:
        detail = output.splitlines()[-1] if output else "No branch information returned."
        log.warning("Could not fetch branches: %s", detail)
        return gr.update(choices=["main"], value="main")

    branches: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if "refs/heads/" in line:
            branches.append(line.rsplit("refs/heads/", 1)[-1])
        else:
            branches.append(line.lstrip("* ").strip())

    branches = sorted({branch for branch in branches if branch})
    if not branches:
        return gr.update(choices=["main"], value="main")

    preferred = "main" if "main" in branches else "master" if "master" in branches else branches[0]
    return gr.update(choices=branches, value=preferred)


def clone_project(repo: str, branch: str, destination: str):
    repo = (repo or "").strip()
    branch = (branch or "").strip()
    if not repo:
        return "Missing repository path or URL."

    dest = _resolve_destination(destination)
    if dest.exists() and any(dest.iterdir()):
        return f"Using existing copy at `{dest}`."

    dest.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = ["git", "clone", "--depth", "1"]
    if branch:
        clone_cmd.extend(["--branch", branch])
    clone_cmd.extend([repo, str(dest)])

    try:
        code, output = _run_command(clone_cmd, timeout=180)
    except Exception as exc:  # noqa: BLE001 - surface setup issue to the UI
        return f"Clone failed: `{exc}`"

    if code != 0:
        detail = output.splitlines()[-1] if output else "No output returned."
        return f"Clone failed: `{detail}`"

    return f"Cloned to `{dest}`."


def _derived_kg_name(repo: str) -> str:
    """Derive a knowledge-graph folder name from a repo URL or path."""
    name = (repo or "").strip()
    # Strip trailing .git and take the last path segment.
    if name.endswith(".git"):
        name = name[:-4]
    # Take the last segment from a URL / path.
    name = name.rstrip("/").rsplit("/", 1)[-1]
    # Sanitise: keep only alphanumeric, dash, underscore.
    import re as _re
    name = _re.sub(r"[^a-zA-Z0-9_-]", "-", name).strip("-") or "repo-kg"
    return name


def clone_and_build_project(repo: str, branch: str, destination: str):
    """Clone a repo, then automatically build its Knowledge Graph locally.

    Returns (clone_status_text, kg_status_text).
    """
    # ── Step 1: Clone ───────────────────────────────────────────────────
    clone_msg = clone_project(repo, branch, destination)
    if "failed" in clone_msg.lower():
        return clone_msg, f"KG build skipped — {clone_msg}"

    dest = _resolve_destination(destination)
    kg_name = _derived_kg_name(repo)
    log.info("Auto-building KG for cloned repo. repo=%s, dest=%s, kg_name=%s",
             repo[:80], dest, kg_name)

    # ── Step 2: Trigger KG build via local backend API ──────────────────
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/knowledge-graph/build",
            json={"source_path": str(dest), "kg_name": kg_name, "upload": False},
            timeout=60 * 60,  # KG builds can take a while
        )
        if resp.status_code == 200:
            data = resp.json()
            stats = data.get("stats", {})
            chunks = stats.get("chunks", "?")
            msg = (
                f"Knowledge Graph built: **{kg_name}** "
                f"({chunks} chunks, {stats.get('nodes', '?')} nodes, "
                f"{stats.get('edges', '?')} edges)"
            )
            log.info("KG build succeeded: %s", msg)
            # Update global STATE so the UI status chip reflects the new KG.
            try:
                retrieval = _get_retrieval_service()
                retrieval.refresh()
                STATE["kgs"] = retrieval.kg_names
            except Exception:
                pass
            return clone_msg, msg
        else:
            err = resp.text[:500]
            log.warning("KG build returned HTTP %d: %s", resp.status_code, err)
            return clone_msg, f"KG build failed — HTTP {resp.status_code}: {err}"
    except requests.exceptions.ConnectionError:
        log.warning("KG build: backend not reachable at %s", BACKEND_URL)
        return clone_msg, "KG build failed — backend not reachable."
    except Exception as exc:  # noqa: BLE001
        log.exception("KG build failed: %s", exc)
        return clone_msg, f"KG build failed — {exc}"


def _get_retrieval_service():
    """Lazily import and return the retrieval service singleton."""
    from api.services.retrieval_service import get_retrieval_service
    return get_retrieval_service(flask_app.config)


def _repo_prompt(repo: str, branch: str, destination: str, problem: str) -> str:
    return "\n".join(
        [
            "You are Rubber Duck Games, a friendly game development debugging assistant.",
            "The user wants you to reason from a duplicated Git project, not from a pasted snippet.",
            "",
            f"Repository: {(repo or '').strip() or '(not provided)'}",
            f"Branch: {(branch or '').strip() or 'main'}",
            f"Local copy: {str(_resolve_destination(destination))}",
            "",
            "Observed problem:",
            (problem or "Help me inspect the project and decide what to test first.").strip(),
            "",
            "Respond ONLY with a single JSON object matching this schema (no markdown, no extra text before or after",
            "the opening/closing braces):",
            "",
            "{",
            '  "schema_version": "1.0",',
            '  "session": {',
            '    "mode": "duck_question",',
            '    "repo": {',
            '      "url": "<repo url>",',
            '      "branch": "<branch>",',
            '      "local_path": "<local path>"',
            "    },",
            '    "user_problem": "<one-line summary of the problem>"',
            "  },",
            '  "conversation": {',
            '    "messages": [',
            "      {",
            '        "id": "q1",',
            '        "role": "duck",',
            '        "kind": "question",',
            '        "content": "<first diagnostic question>",',
            '        "intent": "<why you ask this>",',
            '        "expects_user_reply": true',
            "      },",
            "      {",
            '        "id": "q2", "role": "duck", "kind": "question",',
            '        "content": "<second question>",',
            '        "intent": "<intent>",',
            '        "expects_user_reply": true',
            "      },",
            "      {",
            '        "id": "q3", "role": "duck", "kind": "question",',
            '        "content": "<third question>",',
            '        "intent": "<intent>",',
            '        "expects_user_reply": true',
            "      },",
            "      {",
            '        "id": "q4", "role": "duck", "kind": "question",',
            '        "content": "<fourth question>",',
            '        "intent": "<intent>",',
            '        "expects_user_reply": true',
            "      }",
            "    ],",
            '    "next_prompt_hint": "<a short hint for what the user could try next>"',
            "  },",
            '  "repo_findings": [',
            "    {",
            '      "id": "finding_1",',
            '      "title": "<finding title>",',
            '      "summary": "<one-sentence finding summary>",',
            '      "evidence": [',
            '        {"file": "<relevant file>", "symbol": "<function/class>", "reason": "<why relevant>"}',
            "      ],",
            '      "confidence": "low|medium|high",',
            '      "learning_opportunity": {',
            '        "concept": "<concept name>",',
            '        "why_it_matters": "<explanation>",',
            '        "beginner_explanation": "<beginner-friendly note>",',
            '        "suggested_next_step": "<concrete next action>"',
            "      }",
            "    }",
            "  ],",
            '  "fix_options": [',
            "    {",
            '      "id": "fix_1",',
            '      "area": "input|movement|rendering|physics|audio|other",',
            '      "title": "<fix title>",',
            '      "description": "<what the fix does>",',
            '      "complexity": "low|medium|high",',
            '      "risk": "low|medium|high",',
            '      "recommended": true,',
            '      "steps": ["<step 1>", "<step 2>"],',
            '      "tradeoffs": ["<tradeoff 1>"]',
            "    }",
            "  ],",
            '  "refactor_suggestion": {',
            '    "title": "<refactor title>",',
            '    "reason": "<why refactor>",',
            '    "when_to_do_it": "now|after_fix|later",',
            '    "scope": "<what files/concepts are affected>"',
            "  }",
            "}",
        ]
    )


def _repo_followup_prompt(repo: str, branch: str, destination: str, problem: str, followup: str) -> str:
    """Build a focused follow-up prompt for the next round of duck questions.

    The follow-up only needs to refresh the Duck Questions section, so we reuse
    the small, single-section schema (the same one the initial flow uses) rather
    than asking the model to re-emit the entire session. Requesting the full
    schema here overflows the short token budget, truncates the JSON, and breaks
    the required format — so we keep the ask tight and conformant instead.
    """
    return "\n".join(
        _session_header(repo, branch, destination, problem)
        + [
            "The user is continuing the same debugging conversation.",
            "Their latest reply or observation:",
            (followup or "").strip(),
            "",
            "Based on that reply, return EXACTLY this shape with up to 4 short, updated",
            "diagnostic questions that guide the user closer to the cause:",
            "",
            "{",
            '  "conversation": {',
            '    "messages": [',
            '      {"id": "q1", "role": "duck", "kind": "question", "content": "<first diagnostic question>", "intent": "<why you ask this>", "expects_user_reply": true},',
            '      {"id": "q2", "role": "duck", "kind": "question", "content": "<second question>", "intent": "<intent>", "expects_user_reply": true},',
            '      {"id": "q3", "role": "duck", "kind": "question", "content": "<third question>", "intent": "<intent>", "expects_user_reply": true},',
            '      {"id": "q4", "role": "duck", "kind": "question", "content": "<fourth question>", "intent": "<intent>", "expects_user_reply": true}',
            "    ],",
            '    "next_prompt_hint": "<a short hint for what the user could try next>"',
            "  }",
            "}",
        ]
    )



# ── Per-section prompts (sequential, tab-by-tab generation) ──────────────────
# Generating the full schema in one shot is slow and ties up every tab until
# the very end. Instead we ask the model for one small section at a time, so
# each call is short (fast to generate, stops on a complete JSON object) and the
# corresponding tab can render the moment its section is ready.

def _session_header(repo: str, branch: str, destination: str, problem: str) -> list[str]:
    return [
        "You are Rubber Duck Games, a friendly game development debugging assistant.",
        "You reason from a duplicated Git project, not from a pasted snippet.",
        "",
        f"Repository: {(repo or '').strip() or '(not provided)'}",
        f"Branch: {(branch or '').strip() or 'main'}",
        f"Local copy: {str(_resolve_destination(destination))}",
        "",
        "Observed problem:",
        (problem or "Help me inspect the project and decide what to test first.").strip(),
        "",
        "Respond ONLY with a single JSON object (no markdown, no text before or after the braces).",
    ]


def _duck_questions_prompt(repo: str, branch: str, destination: str, problem: str) -> str:
    return "\n".join(
        _session_header(repo, branch, destination, problem)
        + [
            "Return EXACTLY this shape with up to 4 short diagnostic questions that guide",
            "the user to find the cause themselves:",
            "",
            "{",
            '  "conversation": {',
            '    "messages": [',
            '      {"id": "q1", "role": "duck", "kind": "question", "content": "<first diagnostic question>", "intent": "<why you ask this>", "expects_user_reply": true},',
            '      {"id": "q2", "role": "duck", "kind": "question", "content": "<second question>", "intent": "<intent>", "expects_user_reply": true},',
            '      {"id": "q3", "role": "duck", "kind": "question", "content": "<third question>", "intent": "<intent>", "expects_user_reply": true},',
            '      {"id": "q4", "role": "duck", "kind": "question", "content": "<fourth question>", "intent": "<intent>", "expects_user_reply": true}',
            "    ],",
            '    "next_prompt_hint": "<a short hint for what the user could try next>"',
            "  }",
            "}",
        ]
    )


def _repo_findings_prompt(repo: str, branch: str, destination: str, problem: str) -> str:
    return "\n".join(
        _session_header(repo, branch, destination, problem)
        + [
            "Return EXACTLY this shape with 1-2 concrete findings grounded in the repo's files:",
            "",
            "{",
            '  "repo_findings": [',
            "    {",
            '      "id": "finding_1",',
            '      "title": "<finding title>",',
            '      "summary": "<one-sentence finding summary>",',
            '      "evidence": [{"file": "<relevant file>", "symbol": "<function/class>", "reason": "<why relevant>"}],',
            '      "confidence": "low|medium|high",',
            '      "learning_opportunity": {"concept": "<concept name>", "why_it_matters": "<explanation>", "beginner_explanation": "<beginner-friendly note>", "suggested_next_step": "<concrete next action>"}',
            "    }",
            "  ]",
            "}",
        ]
    )


def _fix_options_prompt(repo: str, branch: str, destination: str, problem: str) -> str:
    return "\n".join(
        _session_header(repo, branch, destination, problem)
        + [
            "Return EXACTLY this shape with 1-2 focused fix options (mark one as recommended):",
            "",
            "{",
            '  "fix_options": [',
            "    {",
            '      "id": "fix_1",',
            '      "area": "input|movement|rendering|physics|audio|other",',
            '      "title": "<fix title>",',
            '      "description": "<what the fix does>",',
            '      "complexity": "low|medium|high",',
            '      "risk": "low|medium|high",',
            '      "recommended": true,',
            '      "steps": ["<step 1>", "<step 2>"],',
            '      "tradeoffs": ["<tradeoff 1>"]',
            "    }",
            "  ]",
            "}",
        ]
    )


def _refactor_prompt(repo: str, branch: str, destination: str, problem: str) -> str:
    return "\n".join(
        _session_header(repo, branch, destination, problem)
        + [
            "Return EXACTLY this shape with a single, small refactor suggestion:",
            "",
            "{",
            '  "refactor_suggestion": {',
            '    "title": "<refactor title>",',
            '    "reason": "<why refactor>",',
            '    "when_to_do_it": "now|after_fix|later",',
            '    "scope": "<what files/concepts are affected>"',
            "  }",
            "}",
        ]
    )



PREVIEW_STRUCTURED_OUTPUT = {
    "schema_version": "1.0",
    "session": {
        "mode": "duck_question",
        "repo": {
            "url": "https://github.com/user/platformer-game",
            "branch": "main",
            "local_path": "./runs/duck-repo-copy",
        },
        "user_problem": "The player does not move when I press the arrow keys.",
    },
    "conversation": {
        "messages": [
            {
                "id": "preview_q1",
                "role": "duck",
                "kind": "question",
                "content": "When you press an arrow key, do you see any input value change if you print the movement vector?",
                "intent": "confirm_input_signal",
                "expects_user_reply": True,
            },
            {
                "id": "preview_q2",
                "role": "duck",
                "kind": "question",
                "content": "Is the movement code running inside the physics update, or only once when the scene loads?",
                "intent": "locate_update_loop",
                "expects_user_reply": True,
            },
            {
                "id": "preview_q3",
                "role": "duck",
                "kind": "question",
                "content": "Does the player node have a collision body that might be blocked immediately at spawn?",
                "intent": "check_collision_blocker",
                "expects_user_reply": True,
            },
        ],
        "next_prompt_hint": "Try one tiny check first: print the input vector while pressing left and right.",
    },
    "repo_findings": [
        {
            "id": "preview_finding_1",
            "title": "Movement depends on named input actions",
            "summary": "The player script appears to read action names, so the Input Map must contain the same names.",
            "evidence": [
                {
                    "file": "player/player.gd",
                    "symbol": "_physics_process",
                    "reason": "Reads left/right actions every frame before applying velocity.",
                },
                {
                    "file": "project.godot",
                    "symbol": "input",
                    "reason": "This is where Godot stores project-level input action bindings.",
                },
            ],
            "confidence": "high",
            "learning_opportunity": {
                "concept": "Input actions",
                "why_it_matters": "Actions let code ask for intent, like move_left, instead of a specific keyboard key.",
                "beginner_explanation": "If the action name in code is not registered in the project settings, pressing the key can look like nothing is happening.",
                "suggested_next_step": "Open Project Settings > Input Map and compare the action names with the player script.",
            },
        },
        {
            "id": "preview_finding_2",
            "title": "The update loop may be the right place to test first",
            "summary": "Movement bugs are easier to isolate when you confirm the frame-by-frame update is actually running.",
            "evidence": [
                {
                    "file": "player/player.gd",
                    "symbol": "_physics_process",
                    "reason": "Expected place for physics movement and collision-aware motion.",
                }
            ],
            "confidence": "medium",
            "learning_opportunity": {
                "concept": "Game loop debugging",
                "why_it_matters": "A movement assignment that runs only once will not respond continuously to input.",
                "beginner_explanation": "Games re-check input many times per second; a single startup check is usually not enough for movement.",
                "suggested_next_step": "Add one temporary print inside the physics update and verify it repeats while the scene runs.",
            },
        },
    ],
    "fix_options": [
        {
            "id": "preview_fix_1",
            "area": "input",
            "title": "Align the Input Map with the player script",
            "description": "Create or rename the missing actions so the code and project settings use the same names.",
            "complexity": "low",
            "risk": "low",
            "recommended": True,
            "steps": [
                "Find the action names read by the player script.",
                "Add matching actions in Project Settings > Input Map.",
                "Bind arrow keys or WASD to those actions.",
                "Run the scene and test one direction at a time.",
            ],
            "tradeoffs": [
                "Smallest change and easiest to verify.",
                "Does not clean up broader movement structure.",
            ],
        },
        {
            "id": "preview_fix_2",
            "area": "movement",
            "title": "Add a temporary movement trace",
            "description": "Print the calculated input vector and velocity before changing gameplay code.",
            "complexity": "low",
            "risk": "low",
            "recommended": False,
            "steps": [
                "Print the input vector inside the physics update.",
                "Press each movement key and watch the output.",
                "Remove the print once the failing step is clear.",
            ],
            "tradeoffs": [
                "Adds temporary noise to the console.",
                "Helps avoid changing the wrong part of the code.",
            ],
        },
    ],
    "refactor_suggestion": {
        "title": "Name movement actions after intent",
        "reason": "Once movement works, action names like move_left and jump will be easier to scan than engine defaults or key-specific names.",
        "when_to_do_it": "after_fix",
        "scope": "Input Map entries and the small section of player movement code that reads them.",
    },
}

PREVIEW_STRUCTURED, _ = normalize_structured_answer(json.dumps(PREVIEW_STRUCTURED_OUTPUT))
PREVIEW_DUCK_QUESTIONS = render_duck_questions(PREVIEW_STRUCTURED)
PREVIEW_REPO_FINDINGS = render_repo_findings(PREVIEW_STRUCTURED)
PREVIEW_FIX_OPTIONS = render_fix_options(PREVIEW_STRUCTURED)
PREVIEW_REFACTOR = render_refactor(PREVIEW_STRUCTURED)
PREVIEW_RAW_RESPONSE = json.dumps(PREVIEW_STRUCTURED, indent=2)


def _render_structured_response(raw_answer: str, raw_json: str, repo: str, branch: str,
                                destination: str, problem: str):
    structured, error = normalize_structured_answer(
        raw_answer,
        repo_url=repo,
        branch=branch,
        local_path=str(_resolve_destination(destination)),
        user_problem=problem,
    )
    raw_payload = json.loads(raw_json) if raw_json and raw_json.strip().startswith("{") else {"raw": raw_json}
    raw_payload["structured_preview_error"] = error
    raw_payload["structured"] = structured
    return (
        render_duck_questions(structured),
        render_repo_findings(structured),
        render_fix_options(structured),
        render_refactor(structured),
        json.dumps(raw_payload, indent=2, ensure_ascii=False),
    )


def _pending_card(title: str) -> str:
    """A small placeholder shown in a tab while its section is still generating."""
    return (
        f'<div class="info-card"><div class="card-kicker">{html.escape(title)}</div>'
        'Generating… the duck is still working on this section.'
        '</div>'
    )


def ask_repo(repo: str, branch: str, destination: str, problem: str,
             use_context: bool, temperature: float):
    """Generate the duck session one tab at a time and stream each as it's ready.

    Rather than producing the whole schema in a single (slow) call that blocks
    every tab until the end, we ask the model for one small section per call.
    Each call is short — and stops the instant it emits a complete JSON object —
    so Duck Questions can render while Repo Findings are still being generated,
    and so on. This is a Gradio generator: each ``yield`` pushes a fresh set of
    tab contents to the UI.
    """
    local_path = str(_resolve_destination(destination))

    # Accumulate the raw section fragments; re-normalize the merged object after
    # each step so every tab renders consistently (with defaults filled in).
    accumulated: dict = {
        "schema_version": "1.0",
        "session": {
            "mode": "duck_question",
            "repo": {"url": repo, "branch": branch, "local_path": local_path},
            "user_problem": problem,
        },
    }

    def _structured() -> dict:
        structured, _ = normalize_structured_answer(
            json.dumps(accumulated),
            repo_url=repo,
            branch=branch,
            local_path=local_path,
            user_problem=problem,
        )
        return structured

    def _raw(structured: dict) -> str:
        return json.dumps(structured, indent=2, ensure_ascii=False)

    sections = [
        ("Duck Questions", _duck_questions_prompt, "duck_questions"),
        ("Repo Findings", _repo_findings_prompt, "repo_findings"),
        ("Fix Options", _fix_options_prompt, "fix_options"),
        ("Refactor", _refactor_prompt, "refactor"),
    ]

    # Initial frame: everything pending so the user sees immediate feedback.
    pending = {
        "duck_questions": _pending_card("Duck Questions"),
        "repo_findings": _pending_card("Repo Findings"),
        "fix_options": _pending_card("Fix Options"),
        "refactor": _pending_card("Refactor"),
    }
    yield (
        pending["duck_questions"],
        pending["repo_findings"],
        pending["fix_options"],
        pending["refactor"],
        "{}",
    )

    rendered = dict(pending)
    for title, prompt_fn, key in sections:
        prompt = prompt_fn(repo, branch, destination, problem)
        answer, _ = ask(prompt, use_context, 384, temperature,
                        stop_on_json=True)
        fragment = parse_partial_json(answer)
        if fragment:
            accumulated.update(fragment)
        structured = _structured()

        # Refresh the tab that just completed; keep later tabs on their pending
        # placeholders so the user can tell what's still being generated.
        rendered["duck_questions"] = render_duck_questions(structured)
        if key in ("repo_findings", "fix_options", "refactor"):
            rendered["repo_findings"] = render_repo_findings(structured)
        if key in ("fix_options", "refactor"):
            rendered["fix_options"] = render_fix_options(structured)
        if key == "refactor":
            rendered["refactor"] = render_refactor(structured)

        yield (
            rendered["duck_questions"],
            rendered["repo_findings"],
            rendered["fix_options"],
            rendered["refactor"],
            _raw(structured),
        )



def continue_repo_conversation(repo: str, branch: str, destination: str, problem: str, followup: str,
                               use_context: bool, temperature: float):
    followup = (followup or "").strip()
    if not followup:
        return gr.update(), ""

    prompt = _repo_followup_prompt(repo, branch, destination, problem, followup)
    answer, raw = ask(prompt, use_context, 384, temperature, stop_on_json=True)
    structured, _ = normalize_structured_answer(
        answer,
        repo_url=repo,
        branch=branch,
        local_path=str(_resolve_destination(destination)),
        user_problem=problem,
    )
    return render_duck_questions(structured), ""


def repo_summary_html(repo: str, branch: str, destination: str) -> str:
    repo_text = html.escape((repo or "No repository selected").strip())
    branch_text = html.escape((branch or "main").strip())
    return f"""
    <div class="info-card">
        <div class="card-kicker">Repo Context</div>
        <strong>{repo_text}</strong>
        <div><span class="repo-pill">{branch_text}</span></div>
    </div>
    """


with gr.Blocks(title="Rubber Duck Games", css=APP_CSS) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        with gr.Row(elem_classes=["app-grid"]):
            with gr.Column(elem_classes=["side-panel"]):
                gr.HTML(
                    """
                    <div class="brand-row">
                        <div class="duck-mark"></div>
                        <div>
                            <div class="brand-title">Rubber Duck Games</div>
                            <div class="brand-subtitle">Clone a project, then debug beside the duck.</div>
                        </div>
                    </div>
                    """
                )
                status = gr.HTML(value=backend_status(), elem_classes=["backend-status"])

                gr.HTML('<div class="section-label">Project Repository</div>')
                repo_input = gr.Textbox(
                    label="Repository URL",
                    value="https://github.com/user/platformer-game",
                    placeholder="https://github.com/user/game.git or /workspace/game",
                    lines=1,
                    elem_classes=["compact-input"],
                )
                fetch_branches_button = gr.Button(
                    "Fetch branches",
                    size="sm",
                    variant="primary",
                    elem_classes=["primary-duck"],
                )
                branch_input = gr.Dropdown(
                    label="Branch",
                    choices=["main"],
                    value="main",
                    allow_custom_value=True,
                    elem_classes=["compact-input"],
                )

                destination_input = gr.State("./runs/duck-repo-copy")
                clone_status = gr.Markdown("")
                kg_status = gr.Markdown("")

                clone_button = gr.Button("Clone Repo & Build KG", variant="primary", elem_classes=["primary-duck"])

                with gr.Accordion("Model options", open=False):
                    use_context = gr.Checkbox(value=True, label="Use knowledge-graph context")
                    temperature = gr.Slider(0.0, 1.5, value=0.65, step=0.05, label="Temperature")

            with gr.Column(elem_classes=["session-panel"]):
                with gr.Column(elem_classes=["conversation"]):
                    gr.HTML(
                        """
                        <div class="session-header">
                            <div>
                                <div class="session-title"><span class="status-dot"></span>Debug session ready</div>
                                <div class="session-subtitle">The duck will inspect the cloned repo before you can ask questions.</div>
                            </div>
                        </div>
                        """
                    )
                    repo_summary = gr.HTML(repo_summary_html("", "main", "./runs/duck-repo-copy"))
                    problem_input = gr.Textbox(
                        label="What are you having trouble with?",
                        placeholder="e.g. I can't move the player with the arrow keys, I've tried...",
                        lines=3,
                        elem_classes=["compact-input"],
                    )
                    problem_submit = gr.Button("Ask the duck", variant="primary", elem_classes=["primary-duck"])

                    with gr.Tabs(selected="duck_questions"):
                        with gr.Tab("Duck Questions", id="duck_questions"):
                            duck_questions = gr.HTML(value=PREVIEW_DUCK_QUESTIONS)
                            followup_input = gr.Textbox(
                                label="Continue the conversation",
                                placeholder="e.g. I printed the vector and it stays (0, 0) when I press left.",
                                lines=2,
                                elem_classes=["compact-input"],
                            )
                            followup_submit = gr.Button(
                                "Reply to the duck",
                                variant="primary",
                                elem_classes=["primary-duck"],
                            )
                        with gr.Tab("Repo Findings"):
                            repo_findings = gr.HTML(value=PREVIEW_REPO_FINDINGS)
                        with gr.Tab("Fix Options"):
                            fix_options = gr.HTML(
                                value=PREVIEW_FIX_OPTIONS,
                                elem_classes=["structured-panel"],
                            )
                        with gr.Tab("Refactor"):
                            refactor = gr.HTML(value=PREVIEW_REFACTOR)

                    with gr.Accordion("Raw response", open=False):
                        raw_json = gr.Code(value=PREVIEW_RAW_RESPONSE, label="JSON", language="json")

    clone_button.click(
        clone_and_build_project,
        inputs=[repo_input, branch_input, destination_input],
        outputs=[clone_status, kg_status],
    )
    fetch_branches_button.click(
        fetch_branches,
        inputs=repo_input,
        outputs=branch_input,
    )
    repo_input.submit(
        fetch_branches,
        inputs=repo_input,
        outputs=branch_input,
    )
    problem_submit.click(
        ask_repo,
        inputs=[
            repo_input,
            branch_input,
            destination_input,
            problem_input,
            use_context,
            temperature,
        ],
        outputs=[duck_questions, repo_findings, fix_options, refactor, raw_json],
    )
    problem_input.submit(
        ask_repo,
        inputs=[
            repo_input,
            branch_input,
            destination_input,
            problem_input,
            use_context,
            temperature,
        ],
        outputs=[duck_questions, repo_findings, fix_options, refactor, raw_json],
    )
    followup_submit.click(
        continue_repo_conversation,
        inputs=[
            repo_input,
            branch_input,
            destination_input,
            problem_input,
            followup_input,
            use_context,
            temperature,
        ],
        outputs=[duck_questions, followup_input],
    )
    followup_input.submit(
        continue_repo_conversation,
        inputs=[
            repo_input,
            branch_input,
            destination_input,
            problem_input,
            followup_input,
            use_context,
            temperature,
        ],
        outputs=[duck_questions, followup_input],
    )
    for component in [repo_input, branch_input]:
        component.change(
            repo_summary_html,
            inputs=[repo_input, branch_input, destination_input],
            outputs=repo_summary,
        )
    demo.load(lambda: backend_status(), outputs=status)
    status_timer = gr.Timer(value=5)
    status_timer.tick(lambda: backend_status(), outputs=status)


if __name__ == "__main__":
    # Give the backend a moment to come up so the first status read is accurate.
    _wait_for_backend(timeout=15)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
