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
import html
import json
import logging
import os
import subprocess
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
.info-card strong {
    color: #243247 !important;
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
    color: #17675f;
}

.tag-blue {
    background: #eaf1ff;
    color: #3561b7;
}

.tag-gold {
    background: #fff0cd;
    color: #94621a;
}

.tag-green {
    background: #e5f7ef;
    color: #24704d;
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
            "Respond with four duck questions first, then likely repo findings, two or three fix options, "
            "and one refactor suggestion. Keep the tone concise and beginner-friendly.",
        ]
    )


def ask_repo(repo: str, branch: str, destination: str, problem: str,
             use_context: bool, max_new_tokens: int, temperature: float):
    prompt = _repo_prompt(repo, branch, destination, problem)
    answer, raw = ask(prompt, use_context, max_new_tokens, temperature)
    return gr.update(visible=True), answer, raw


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

                clone_button = gr.Button("Clone project", variant="primary", elem_classes=["primary-duck"])

                with gr.Accordion("Model options", open=False):
                    use_context = gr.Checkbox(value=True, label="Use knowledge-graph context")
                    max_new_tokens = gr.Slider(64, 2048, value=768, step=64, label="Max new tokens")
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

                    with gr.Group(visible=False) as results_panel:
                        with gr.Tabs(selected="duck_questions"):
                            with gr.Tab("Duck Questions", id="duck_questions"):
                                gr.HTML(
                                    """
                                    <div class="chat-thread">
                                        <div class="chat-line">
                                            <div class="avatar">D</div>
                                            <div class="chat-card">
                                                <div class="card-kicker">Duck</div>
                                                <ol>
                                                    <li>...?</li>
                                                </ol>
                                            </div>
                                        </div>
                                        <div class="chat-line">
                                            <div class="avatar">You</div>
                                            <div class="chat-card">
                                                <div class="card-kicker">Your turn</div>
                                                Describe the broken behavior in one or two sentences.
                                            </div>
                                        </div>
                                    </div>
                                    """
                                )
                            with gr.Tab("Repo Findings"):
                                gr.HTML(
                                    """
                                    <div class="info-card">
                                        <div class="card-kicker">Repo Findings</div>
                                        Clone the project to give the duck file-level context for its suggestions.
                                        <div><span class="repo-pill">Reveal 2-3 possible fixes</span></div>
                                    </div>
                                    """
                                )
                            with gr.Tab("Fix Options"):
                                answer = gr.Markdown("Ask the duck to generate focused fix options from the repo context.")
                            with gr.Tab("Refactor"):
                                gr.HTML(
                                    """
                                    <div class="stack-list">
                                        <div class="info-card">
                                            <div class="card-kicker">Refactor Suggestion</div>
                                            Prefer a small, testable change first. Once the failing behavior is pinned down,
                                            the duck will suggest whether the surrounding system deserves a cleanup.
                                        </div>
                                        <div class="info-card">
                                            <div class="card-kicker">Confidence</div>
                                            Likely repo/test ownership issue
                                            <div class="confidence-bar"><div></div></div>
                                            <span class="tag tag-blue">repo</span>
                                            <span class="tag tag-green">tests</span>
                                            <span class="tag tag-gold">race risk</span>
                                        </div>
                                    </div>
                                    """
                                )

                        with gr.Accordion("Raw response", open=False):
                            raw_json = gr.Code(label="JSON", language="json")

    clone_button.click(
        clone_project,
        inputs=[repo_input, branch_input, destination_input],
        outputs=clone_status,
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
            max_new_tokens,
            temperature,
        ],
        outputs=[results_panel, answer, raw_json],
    )
    problem_input.submit(
        ask_repo,
        inputs=[
            repo_input,
            branch_input,
            destination_input,
            problem_input,
            use_context,
            max_new_tokens,
            temperature,
        ],
        outputs=[results_panel, answer, raw_json],
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
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
