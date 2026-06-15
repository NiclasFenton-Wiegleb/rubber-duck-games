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
import faulthandler
import logging
import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

# Keep local native math/tokenizer libraries from spawning competing threadpools.
# This makes native crashes easier to diagnose and avoids common macOS OpenMP issues.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

faulthandler.enable(all_threads=True)

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
    category=UserWarning,
    module=r"multiprocessing\.resource_tracker",
)

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


# ── Block bitsandbytes on ZeroGPU (prevents the forked-worker CUDA crash) ─────
# bitsandbytes initialises a *real* CUDA driver context the moment it is
# imported (native cuInit/cudaGetDeviceCount via ctypes — it bypasses the
# `spaces` torch-CUDA emulation, so the host tripwires above never see it).
# transformers / accelerate import bitsandbytes while loading the model in this
# host process to probe `is_bitsandbytes_available()`. Once a CUDA context
# exists in the host, every *forked* @spaces.GPU worker inherits a broken CUDA
# state and dies in `worker_init` → `torch.init()` with
# "RuntimeError: No CUDA GPUs are available".
#
# We never use 4-bit on ZeroGPU (llm_service loads in bfloat16 there), so we
# defensively poison the import slot: `import bitsandbytes` now raises
# ImportError, which transformers/accelerate handle gracefully as "not
# available". This is a belt-and-suspenders guard on top of bitsandbytes being
# absent from the Space's requirements.txt — it neutralises stale build caches
# or transitive re-installs. No-op off ZeroGPU.
if _SPACES_AVAILABLE:
    sys.modules.setdefault("bitsandbytes", None)
    log.info("bitsandbytes import blocked on ZeroGPU host (forked-worker CUDA safety).")


# ── Host-side CUDA-init tripwire (ZeroGPU diagnostics) ───────────────────────
# On ZeroGPU the host process must NEVER initialise a real CUDA context: each
# ``@spaces.GPU`` worker is *forked* from this process, and if CUDA was already
# initialised here the worker dies in ``worker_init`` with
# "RuntimeError: No CUDA GPUs are available" — regardless of where the model
# lives. The failure trace points only at the spaces wrapper, never at the line
# in *our* code that actually poisoned CUDA, which makes it very hard to debug.
#
# This tripwire wraps torch's lazy CUDA initialiser so the *first* real CUDA
# init in the host is logged with a full Python stack trace, pinpointing the
# exact call that touched CUDA outside a ``@spaces.GPU`` worker. It is a no-op
# off ZeroGPU.
def _install_cuda_init_tripwire() -> None:
    import traceback

    try:
        import torch
    except Exception:  # torch not importable yet — nothing to guard
        return

    cuda = torch.cuda
    if getattr(cuda, "_rdg_tripwire_installed", False):
        return

    import os as _os

    _host_pid = _os.getpid()

    # ``torch.cuda.device_count()`` is memoised (lru_cache / a module global) the
    # first time it runs. On a ZeroGPU host there is no visible GPU, so that first
    # call caches **0** — and because each @spaces.GPU worker is *forked* from the
    # host, every worker inherits the cached 0 and dies in ``worker_init`` with
    # "No CUDA GPUs are available", even though ``spaces`` assigned it a real GPU.
    # We trace the *first host-side* call so the poisoning is visible in the logs;
    # calls from inside a forked worker (different PID) are ignored.
    _orig_device_count = cuda.device_count

    def _traced_device_count(*args, **kwargs):
        result = _orig_device_count(*args, **kwargs)
        if _os.getpid() == _host_pid and not getattr(cuda, "_rdg_dc_logged", False):
            cuda._rdg_dc_logged = True
            log.warning(
                "HOST torch.cuda.device_count() -> %s (first host call). If this "
                "is 0 it will be cached and inherited by the forked ZeroGPU "
                "worker, breaking GPU init. Caller:\n%s",
                result, "".join(traceback.format_stack()),
            )
        return result

    cuda.device_count = _traced_device_count

    cuda._rdg_tripwire_installed = True
    log.info("Host CUDA device-count tripwire installed (ZeroGPU diagnostics).")


def _reset_host_cuda_cache() -> None:
    """Clear torch's memoised CUDA device count in the *host* before a worker fork.

    On ZeroGPU the host has no visible GPU, so any earlier query cached a device
    count of 0. The @spaces.GPU worker is forked from the host and inherits that
    cache, then fails in ``worker_init`` with "No CUDA GPUs are available". By
    clearing the cache here — in the host, right before the worker is spawned —
    the freshly forked worker re-queries CUDA (with its assigned GPU now visible)
    instead of reusing the poisoned 0. No-op off ZeroGPU / if torch is absent.
    """
    try:
        import torch
    except Exception:
        return
    cuda = torch.cuda
    # torch >= 2.x: device_count is wrapped with functools.lru_cache.
    dc = getattr(cuda, "device_count", None)
    cache_clear = getattr(dc, "cache_clear", None)
    if callable(cache_clear):
        try:
            cache_clear()
        except Exception:
            pass
    # Some torch builds memoise via a module-level global instead.
    for attr in ("_cached_device_count", "_device_count"):
        if hasattr(cuda, attr):
            try:
                setattr(cuda, attr, None)
            except Exception:
                pass


if _SPACES_AVAILABLE:
    _install_cuda_init_tripwire()



# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

# Make the backend package importable (config.py, api/...).
sys.path.insert(0, str(BACKEND_DIR))

from api.services.structured_output import (  # noqa: E402
    normalize_structured_answer,
    parse_partial_json,
    render_duck_questions,
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

    if "LLM_DEVICE" not in os.environ:
        if _SPACES_AVAILABLE:
            log.info("ZeroGPU runtime detected — setting LLM_DEVICE=cuda; the "
                     "model is eager-loaded onto CPU in the host process (the "
                     "host never touches CUDA) and moved onto the real GPU "
                     "inside each @spaces.GPU worker on first generation.")
            os.environ["LLM_DEVICE"] = "cuda"

        else:
            try:
                import torch  # noqa: F811 (already imported above via spaces?)
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

    if "LLM_LAZY_LOAD" not in os.environ:
        gpu_target = _SPACES_AVAILABLE or os.environ.get("LLM_DEVICE") == "cuda"
        os.environ["LLM_LAZY_LOAD"] = "false" if gpu_target else "true"



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
      * ZeroGPU: the weights are eager-loaded onto **CPU** in this host process
        (the host must never create a CUDA context, or the forked @spaces.GPU
        worker dies in ``worker_init`` with "No CUDA GPUs are available"). Each
        forked worker inherits the CPU weights cheaply (copy-on-write) and moves
        them onto the real GPU on the first generation, inside the worker, where
        CUDA is actually available.

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

.loading-card {
    align-items: flex-start;
    display: flex;
    gap: 10px;
}

.loading-card .build-spinner {
    margin-top: 1px;
}

.loading-card-body {
    display: grid;
    gap: 4px;
}

.loading-card-text {
    color: #243247 !important;
    font-size: 12px;
    font-weight: 650;
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

.build-progress {
    background: #ffffff;
    border: 1px solid #cfd9e5;
    border-radius: 8px;
    color: #172033;
    display: grid;
    gap: 12px;
    margin: 0;
    padding: 12px;
}

.build-progress.is-idle {
    display: none;
}

.build-progress.is-loading {
    background: #fff8df;
    border-color: #d7aa2c;
}

.build-progress.is-success {
    background: #e9f8f4;
    border-color: #82cdbf;
}

.build-progress.is-error {
    background: #fff0f0;
    border-color: #e19494;
}

.build-progress-head {
    align-items: flex-start;
    display: flex;
    gap: 10px;
}

.build-spinner {
    animation: rdg-spin 0.9s linear infinite;
    border: 3px solid rgba(128, 83, 10, 0.24);
    border-top-color: #d9480f;
    border-radius: 50%;
    flex: 0 0 auto;
    height: 19px;
    margin-top: 1px;
    width: 19px;
}

.build-progress.is-success .build-spinner,
.build-progress.is-error .build-spinner {
    animation: none;
    border: 2px solid currentColor;
    border-radius: 50%;
    color: currentColor;
    display: grid;
    height: 19px;
    place-items: center;
    width: 19px;
}

.build-progress.is-success .build-spinner::before {
    border-bottom: 2px solid currentColor;
    border-left: 2px solid currentColor;
    content: "";
    height: 4px;
    transform: rotate(-45deg) translate(1px, -1px);
    width: 8px;
}

.build-progress.is-error .build-spinner::before {
    content: "!";
    font-size: 12px;
    font-weight: 900;
    line-height: 1;
}

.build-progress-title {
    color: #111827;
    font-size: 13px;
    font-weight: 850;
    line-height: 1.25;
}

.build-progress-detail {
    color: #2d3c50;
    font-size: 12px;
    font-weight: 650;
    line-height: 1.45;
    margin-top: 2px;
}

.build-meta {
    background: rgba(255, 255, 255, 0.68);
    border: 1px solid rgba(64, 81, 106, 0.16);
    border-radius: 6px;
    display: grid;
    gap: 7px;
    padding: 9px;
}

.build-meta-row {
    display: grid;
    gap: 3px;
}

.build-meta-label {
    color: #40516a;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.build-meta-value {
    color: #172033;
    font-size: 12px;
    font-weight: 750;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.build-path {
    background: #172033;
    border-radius: 5px;
    color: #ffffff;
    display: block;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    font-weight: 750;
    line-height: 1.35;
    padding: 7px 8px;
    white-space: normal;
    word-break: break-word;
}

.build-steps {
    display: grid;
    gap: 6px;
}

.build-step {
    align-items: center;
    color: #40516a;
    display: flex;
    font-size: 12px;
    font-weight: 750;
    gap: 8px;
}

.build-step-dot {
    background: #aebdcb;
    border-radius: 50%;
    flex: 0 0 auto;
    height: 8px;
    width: 8px;
}

.build-step.done {
    color: #0b5f58;
}

.build-step.done .build-step-dot {
    background: #0f8f80;
}

.build-step.active {
    color: #7a4300;
}

.build-step.active .build-step-dot {
    background: #d9480f;
    box-shadow: 0 0 0 4px rgba(217, 72, 15, 0.16);
}

@keyframes rdg-spin {
    to { transform: rotate(360deg); }
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
                  temperature: float, stop_on_json: bool = False,
                  prepared_context: str = "",
                  prepared_sources_json: str = "[]",
                  retrieval_kg_name: str = "") -> int:
    """Reserve more quota only for the (one-off) first call that loads weights."""
    try:
        from api.services.llm_service import get_llm_service

        if not get_llm_service(flask_app.config).is_loaded:
            return GPU_LOAD_DURATION
    except Exception:
        return GPU_LOAD_DURATION
    return GPU_DURATION


def _run_generation(message: str, use_context: bool, max_new_tokens: int,
                    temperature: float, stop_on_json: bool = False,
                    prepared_context: str = "",
                    prepared_sources_json: str = "[]",
                    retrieval_kg_name: str = "") -> dict:
    """Core generation logic — called directly or wrapped by @spaces.GPU."""
    from api.services.llm_service import get_llm_service

    log.info("Generation starting: message_len=%d, use_context=%s, "
             "max_new_tokens=%d, temperature=%.2f, stop_on_json=%s",
             len(message), use_context, max_new_tokens, temperature, stop_on_json)

    t0 = time.time()
    llm = get_llm_service(flask_app.config)
    prepared_sources: list[dict] = []
    if prepared_context:
        try:
            decoded_sources = json.loads(prepared_sources_json or "[]")
            if isinstance(decoded_sources, list):
                prepared_sources = decoded_sources
        except json.JSONDecodeError:
            log.warning("Could not decode prepared retrieval sources; ignoring")

    result = llm.generate_simple(
        message,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        use_context=bool(use_context),
        stop_on_json=bool(stop_on_json),
        context_override=prepared_context if prepared_context else None,
        sources_override=prepared_sources,
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
        stop_on_json: bool = False, retrieval_query: str | None = None,
        retrieval_kg_name: str | None = None):
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

    generation_use_context = bool(use_context)
    prepared_context = ""
    prepared_sources_json = "[]"
    explicit_retrieval_query = (retrieval_query or "").strip()
    retrieval_input = explicit_retrieval_query or message
    requested_kg = (retrieval_kg_name or "").strip()

    # Retrieval loads SentenceTransformer/FAISS native libraries. On ZeroGPU,
    # doing that inside the forked @spaces.GPU worker has caused hard segfaults
    # just before LLM generation starts, so prepare KG context in the host and
    # pass the plain text/sources into the worker. DuckChat also passes a
    # focused repo/problem retrieval query so the search is based on the copied
    # project, not on the JSON schema instructions in the generation prompt.
    if generation_use_context and (_SPACES_AVAILABLE or explicit_retrieval_query):
        try:
            from api.services.llm_service import get_llm_service

            llm = get_llm_service(flask_app.config)
            sources: list[dict]
            prepared_context, sources = llm.prepare_context(
                retrieval_input,
                True,
                kg_names=[requested_kg] if requested_kg else None,
            )
            prepared_sources_json = json.dumps(sources, ensure_ascii=False)
            generation_use_context = False
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not prepare retrieval context before GPU worker: %s", exc)
            generation_use_context = False

    # Clear torch's host-side CUDA device-count cache right before the ZeroGPU
    # worker is forked, so the worker re-detects its freshly-assigned GPU instead
    # of inheriting the host's cached "0 devices" (the cause of the worker dying
    # in worker_init with "No CUDA GPUs are available"). No-op off ZeroGPU.
    if _SPACES_AVAILABLE:
        _reset_host_cuda_cache()

    try:
        result = _generate_simple(
            message,
            generation_use_context,
            max_new_tokens,
            temperature,
            stop_on_json,
            prepared_context,
            prepared_sources_json,
            requested_kg,
        )

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


def clone_project(repo: str, branch: str, destination: str) -> dict:
    repo = (repo or "").strip()
    branch = (branch or "").strip()
    if not repo:
        return {"ok": False, "message": "Missing repository path or URL.", "path": "", "reused": False}

    dest = _resolve_destination(destination)
    if dest.exists() and any(dest.iterdir()):
        return {
            "ok": True,
            "message": "Using existing local copy.",
            "path": str(dest),
            "reused": True,
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = ["git", "clone", "--depth", "1"]
    if branch:
        clone_cmd.extend(["--branch", branch])
    clone_cmd.extend([repo, str(dest)])

    try:
        code, output = _run_command(clone_cmd, timeout=180)
    except Exception as exc:  # noqa: BLE001 - surface setup issue to the UI
        return {"ok": False, "message": f"Clone failed: {exc}", "path": str(dest), "reused": False}

    if code != 0:
        detail = output.splitlines()[-1] if output else "No output returned."
        return {"ok": False, "message": f"Clone failed: {detail}", "path": str(dest), "reused": False}

    return {"ok": True, "message": "Cloned repository.", "path": str(dest), "reused": False}


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


def clone_build_progress_html(
    state: str = "idle",
    detail: str = "",
    active_step: str = "clone",
    title: str | None = None,
    repo_path: str = "",
    kg_name: str = "",
    stats: dict | None = None,
) -> str:
    steps = [("clone", "Prepare repository"), ("kg", "Build knowledge graph")]
    step_html = []
    for key, label in steps:
        if state == "success" or (active_step == "kg" and key == "clone"):
            status = "done"
        elif state == "error" and active_step == "kg" and key == "clone":
            status = "done"
        elif key == active_step and state == "loading":
            status = "active"
        else:
            status = ""
        step_html.append(
            f'<div class="build-step {status}"><span class="build-step-dot"></span>{html.escape(label)}</div>'
        )

    titles = {
        "idle": "",
        "loading": "Preparing project",
        "success": "Project ready",
        "error": "Build needs attention",
    }
    meta_rows = []
    if repo_path:
        meta_rows.append(
            '<div class="build-meta-row">'
            '<div class="build-meta-label">Local copy</div>'
            f'<code class="build-path">{html.escape(repo_path)}</code>'
            '</div>'
        )
    if kg_name:
        meta_rows.append(
            '<div class="build-meta-row">'
            '<div class="build-meta-label">Knowledge graph</div>'
            f'<div class="build-meta-value">{html.escape(kg_name)}</div>'
            '</div>'
        )
    if stats:
        stats_text = (
            f"{stats.get('chunks', '?')} chunks, "
            f"{stats.get('nodes', '?')} nodes, "
            f"{stats.get('edges', '?')} edges"
        )
        meta_rows.append(
            '<div class="build-meta-row">'
            '<div class="build-meta-label">Indexed</div>'
            f'<div class="build-meta-value">{html.escape(stats_text)}</div>'
            '</div>'
        )

    safe_state = html.escape(state)
    safe_detail = html.escape(detail)
    safe_title = html.escape(title or titles.get(state, titles["loading"]))
    meta_html = f'<div class="build-meta">{"".join(meta_rows)}</div>' if meta_rows else ""
    return (
        f'<div class="build-progress is-{safe_state}" role="status" aria-live="polite">'
        '<div class="build-progress-head">'
        '<span class="build-spinner" aria-hidden="true"></span>'
        '<div>'
        f'<div class="build-progress-title">{safe_title}</div>'
        f'<div class="build-progress-detail">{safe_detail}</div>'
        '</div>'
        '</div>'
        f'{meta_html}'
        f'<div class="build-steps">{"".join(step_html)}</div>'
        '</div>'
    )


def _ask_duck_button_update(project_ready: bool):
    return gr.update(interactive=bool(project_ready))


def _problem_input_update(project_ready: bool):
    return gr.update(interactive=bool(project_ready))


def clone_and_build_project(repo: str, branch: str, destination: str):
    yield (
        clone_build_progress_html(
            "loading",
            "Step 1 of 2: checking the destination and cloning the selected branch if needed.",
            "clone",
            title="Preparing repository",
        ),
        _ask_duck_button_update(False),
        _problem_input_update(False),
        False,
    )

    clone_result = clone_project(repo, branch, destination)
    if not clone_result["ok"]:
        yield (
            clone_build_progress_html(
                "error",
                clone_result["message"],
                "clone",
                repo_path=clone_result.get("path", ""),
            ),
            _ask_duck_button_update(False),
            _problem_input_update(False),
            False,
        )
        return

    dest = _resolve_destination(destination)
    kg_name = _derived_kg_name(repo)
    log.info("Auto-building KG for cloned repo. repo=%s, dest=%s, kg_name=%s",
             repo[:80], dest, kg_name)

    clone_detail = (
        "Found an existing local copy, so cloning was skipped. "
        if clone_result["reused"]
        else "Repository cloned successfully. "
    )
    yield (
        clone_build_progress_html(
            "loading",
            clone_detail + "Step 2 of 2: scanning files, chunking source, and creating graph links. This can take a few minutes.",
            "kg",
            title="Building knowledge graph",
            repo_path=clone_result["path"],
            kg_name=kg_name,
        ),
        _ask_duck_button_update(False),
        _problem_input_update(False),
        False,
    )

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/knowledge-graph/build",
            json={"source_path": str(dest), "kg_name": kg_name, "upload": False},
            timeout=60 * 60,  # KG builds can take a while
        )
        if resp.status_code == 200:
            data = resp.json()
            stats = data.get("stats", {})
            log.info("KG build succeeded: %s", stats)
            try:
                retrieval = _get_retrieval_service()
                retrieval.refresh()
                STATE["kgs"] = retrieval.kg_names
            except Exception:
                pass
            final_title = "Existing copy indexed" if clone_result["reused"] else "Repo cloned and indexed"
            final_detail = "The knowledge graph is ready. You can ask the duck about this project now."
            yield (
                clone_build_progress_html(
                    "success",
                    final_detail,
                    "kg",
                    title=final_title,
                    repo_path=clone_result["path"],
                    kg_name=kg_name,
                    stats=stats,
                ),
                _ask_duck_button_update(True),
                _problem_input_update(True),
                True,
            )
        else:
            err = resp.text[:500]
            log.warning("KG build returned HTTP %d: %s", resp.status_code, err)
            yield (
                clone_build_progress_html(
                    "error",
                    f"The backend returned HTTP {resp.status_code}: {err}",
                    "kg",
                    repo_path=clone_result["path"],
                    kg_name=kg_name,
                ),
                _ask_duck_button_update(False),
                _problem_input_update(False),
                False,
            )
    except requests.exceptions.ConnectionError:
        log.warning("KG build: backend not reachable at %s", BACKEND_URL)
        yield (
            clone_build_progress_html(
                "error",
                "The backend is not reachable, so the knowledge graph could not be built.",
                "kg",
                repo_path=clone_result["path"],
                kg_name=kg_name,
            ),
            _ask_duck_button_update(False),
            _problem_input_update(False),
            False,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("KG build failed: %s", exc)
        yield (
            clone_build_progress_html(
                "error",
                f"Knowledge graph build failed: {exc}",
                "kg",
                repo_path=clone_result["path"],
                kg_name=kg_name,
            ),
            _ask_duck_button_update(False),
            _problem_input_update(False),
            False,
        )


def _get_retrieval_service():
    """Lazily import and return the retrieval service singleton."""
    from api.services.retrieval_service import get_retrieval_service
    return get_retrieval_service(flask_app.config)


def _repo_followup_prompt(repo: str, branch: str, destination: str, problem: str, followup: str) -> str:
    """Build a focused follow-up prompt for the next round of duck questions.

    The follow-up only needs to refresh the Quacking section, so we reuse
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
            "Based on that reply, return EXACTLY this shape with exactly 1 short, updated",
            "diagnostic question that guides the user closer to the cause:",
            "",
            "{",
            '  "conversation": {',
            '    "messages": [',
            '      {"id": "q1", "role": "duck", "kind": "question", "content": "<diagnostic question>", "intent": "<why you ask this>", "expects_user_reply": true}',
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
            "Return EXACTLY this shape with exactly 1 short diagnostic question that guides the developer to find the cause themselves:",
            "",
            "{",
            '  "conversation": {',
            '    "messages": [',
            '      {"id": "q1", "role": "duck", "kind": "question", "content": "<diagnostic question>", "intent": "<why you ask this>", "expects_user_reply": true}',
            "    ],",
            '    "next_prompt_hint": "<a short hint for what the user could try next>"',
            "  }",
            "}",
        ]
    )


def _repo_retrieval_query(repo: str, branch: str, destination: str, problem: str) -> str:
    return "\n".join(
        [
            f"Repository: {(repo or '').strip() or '(not provided)'}",
            f"Branch: {(branch or '').strip() or 'main'}",
            f"Knowledge graph: {_derived_kg_name(repo)}",
            f"Local copy: {str(_resolve_destination(destination))}",
            "",
            "User debugging problem:",
            (problem or "Help me inspect the project and decide what to test first.").strip(),
            "",
            "Retrieve source files, project settings, input mappings, scene scripts, physics code, and nearby symbols relevant to this problem.",
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
        ],
        "next_prompt_hint": "Try one tiny check first: print the input vector while pressing left and right.",
    },
}

PREVIEW_STRUCTURED, _ = normalize_structured_answer(json.dumps(PREVIEW_STRUCTURED_OUTPUT))
PREVIEW_DUCK_QUESTIONS = render_duck_questions(PREVIEW_STRUCTURED)
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
    display_structured = dict(structured)
    display_structured.pop("repo_findings", None)
    display_structured.pop("fix_options", None)
    display_structured.pop("refactor_suggestion", None)
    raw_payload["structured"] = display_structured
    return (
        render_duck_questions(structured),
        json.dumps(raw_payload, indent=2, ensure_ascii=False),
    )


def _pending_card(title: str) -> str:
    """A small placeholder shown in a tab while its section is still generating."""
    return (
        '<div class="info-card loading-card">'
        '<span class="build-spinner" aria-hidden="true"></span>'
        '<div class="loading-card-body">'
        f'<div class="card-kicker">{html.escape(title)}</div>'
        '<div class="loading-card-text">Generating response from your favourite duck...</div>'
        '</div>'
        '</div>'
    )


def _ask_first_card(title: str) -> str:
    return (
        f'<div class="info-card"><div class="card-kicker">{html.escape(title)}</div>'
        'Ask a question to get some feedback from your favourite duck.'
        '</div>'
    )


def ask_repo(repo: str, branch: str, destination: str, problem: str,
             use_context: bool, temperature: float):
    """Generate the duck session one tab at a time and stream each as it's ready.

    Rather than producing the whole schema in a single (slow) call that blocks
    every tab until the end, we ask the model for one small section per call.
    Each call is short — and stops the instant it emits a complete JSON object —
    so DuckChat can render as soon as the model returns a question. This is a
    Gradio generator: each ``yield`` pushes a fresh tab content update to the UI.
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
        display_structured = dict(structured)
        display_structured.pop("repo_findings", None)
        display_structured.pop("fix_options", None)
        display_structured.pop("refactor_suggestion", None)
        return json.dumps(display_structured, indent=2, ensure_ascii=False)

    sections = [
        ("Quacking", _duck_questions_prompt, "duck_questions"),
    ]
    kg_name = _derived_kg_name(repo)
    retrieval_query = _repo_retrieval_query(repo, branch, destination, problem)

    # Initial frame: everything pending so the user sees immediate feedback.
    pending = {
        "duck_questions": _pending_card("Quacking"),
    }
    yield (
        pending["duck_questions"],
        "{}",
    )

    rendered = dict(pending)
    for title, prompt_fn, key in sections:
        prompt = prompt_fn(repo, branch, destination, problem)
        answer, _ = ask(
            prompt,
            bool(use_context),
            384,
            temperature,
            stop_on_json=True,
            retrieval_query=retrieval_query if use_context else None,
            retrieval_kg_name=kg_name if use_context else None,
        )
        fragment = parse_partial_json(answer)
        if fragment:
            accumulated.update(fragment)
        structured = _structured()

        # Refresh the tab that just completed; keep later tabs on their pending
        # placeholders so the user can tell what's still being generated.
        rendered["duck_questions"] = render_duck_questions(structured)

        yield (
            rendered["duck_questions"],
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


def repo_context_changed(repo: str, branch: str, destination: str):
    return (
        repo_summary_html(repo, branch, destination),
        clone_build_progress_html(),
        _ask_duck_button_update(False),
        _problem_input_update(False),
        False,
    )


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
                            <div class="brand-subtitle">Clone a project, then debug with the duck!</div>
                        </div>
                    </div>
                    """
                )
                status = gr.HTML(value=backend_status(), elem_classes=["backend-status"])

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
                project_ready = gr.State(False)
                clone_button = gr.Button("Clone Repo & Build KG", variant="primary", elem_classes=["primary-duck"])
                clone_status = gr.HTML(value=clone_build_progress_html())

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
                        interactive=False,
                    )
                    problem_submit = gr.Button(
                        "Ask the duck",
                        variant="primary",
                        elem_classes=["primary-duck"],
                        interactive=False,
                    )

                    with gr.Tabs(selected="duck_questions"):
                        with gr.Tab("DuckChat", id="duck_questions"):
                            duck_questions = gr.HTML(value=_ask_first_card("Duck Chat"))

                    with gr.Accordion("Raw response", open=False):
                        raw_json = gr.Code(value="{}", label="JSON", language="json")

    clone_button.click(
        clone_and_build_project,
        inputs=[repo_input, branch_input, destination_input],
        outputs=[clone_status, problem_submit, problem_input, project_ready],
        show_progress="hidden",
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
        outputs=[duck_questions, raw_json],
        show_progress="hidden",
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
        outputs=[duck_questions, raw_json],
        show_progress="hidden",
    )
    for component in [repo_input, branch_input]:
        component.change(
            repo_context_changed,
            inputs=[repo_input, branch_input, destination_input],
            outputs=[repo_summary, clone_status, problem_submit, problem_input, project_ready],
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
