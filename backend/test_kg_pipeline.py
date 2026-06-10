"""
Debuggable end-to-end test script for the Knowledge Graph pipeline.

What it does
------------
1. Spins up the backend Flask app (in-process, same Python process so that
   breakpoints inside `knowledge_graph_service.py` are actually hit).
2. Takes a local filepath you pass on the command line and feeds it to the
   `POST /api/knowledge-graph/build` endpoint as `source_path`.
3. The endpoint builds the Knowledge Graph (chunks → NetworkX graph → FAISS
   index → manifest) and uploads every artifact to HuggingFace storage. By
   default this is a HuggingFace **Storage Bucket**
   (https://huggingface.co/buckets/<id>); set HF_STORAGE_BACKEND=repo in
   backend/.env to commit to the Space git repo instead.
4. When the build finishes, the script deletes every uploaded artifact back
   off HuggingFace again (clean-up), so you don't leave test data behind.

Why it's debuggable
-------------------
The Flask server runs in a background thread *inside this same process* with
the reloader disabled, so you can set breakpoints anywhere in the request
handler / service code and they will be hit when the request is made. Run this
file directly under the VS Code "Python: Current File" debugger (or
`python -m debugpy`) and step straight through the pipeline.

Usage
-----
    # From the backend/ directory (so config + imports resolve):
    cd backend

    # Build + upload to the bucket + auto-cleanup afterwards:
    python test_kg_pipeline.py "C:/path/to/folder/to/ingest"

    # Build + upload but KEEP the artifacts on HuggingFace (no cleanup):
    python test_kg_pipeline.py "C:/path/to/folder" --no-cleanup

    # Build locally only, never touch HuggingFace:
    python test_kg_pipeline.py "C:/path/to/folder" --no-upload

    # Skip the HTTP layer and call the service directly (easiest stepping):
    python test_kg_pipeline.py "C:/path/to/folder" --direct

Requirements
------------
Uploading / deleting requires a valid HF_TOKEN in backend/.env (see config.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

# Make sure `config` and `api...` import the same way the Flask app does.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# In-process Flask server (so debugger breakpoints in the service get hit)
# ─────────────────────────────────────────────────────────────────────────────
class FlaskServerThread(threading.Thread):
    """Runs the Flask app in a background thread we can cleanly shut down."""

    def __init__(self, app, host: str, port: int):
        super().__init__(daemon=True)
        from werkzeug.serving import make_server

        # threaded=True so the /health probe and the /build request can be
        # served; debug/reloader stay OFF so everything runs in this process.
        self._server = make_server(host, port, app, threaded=True)
        self._ctx = app.app_context()
        self._ctx.push()

    def run(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()


def _wait_for_health(base_url: str, timeout: float = 15.0) -> None:
    """Block until GET /health responds (or raise on timeout)."""
    import requests

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.ok:
                print(f"[server] healthy → {resp.json()}")
                return
        except Exception as exc:  # server not up yet
            last_err = exc
        time.sleep(0.25)
    raise RuntimeError(f"Flask server did not become healthy in time: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace reporting + cleanup (bucket- and repo-aware)
# ─────────────────────────────────────────────────────────────────────────────
def _storage_url(storage: dict) -> str:
    """Human-facing URL where the uploaded artifacts can be viewed."""
    backend = storage.get("backend")
    sid = storage.get("id")
    if backend == "bucket":
        return f"https://huggingface.co/buckets/{sid}"
    repo_type = storage.get("repo_type", "space")
    prefix = {"space": "spaces", "dataset": "datasets"}.get(repo_type, "")
    return f"https://huggingface.co/{prefix + '/' if prefix else ''}{sid}/tree/main"


def report_uploaded(storage: dict | None, uploaded_paths: list[str]) -> None:
    """Tell the user exactly where the uploaded artifacts can be viewed."""
    if not uploaded_paths or not storage:
        return
    backend = storage.get("backend", "?")
    print(f"[upload] {len(uploaded_paths)} artifact(s) pushed to "
          f"{backend}:{storage.get('id')}")
    print(f"[upload] View them here: {_storage_url(storage)}")


def cleanup_hf_artifacts(config, storage: dict | None,
                         uploaded_paths: list[str]) -> None:
    """Delete every uploaded artifact back off HuggingFace (bucket or repo)."""
    if not uploaded_paths:
        print("[cleanup] Nothing was uploaded — nothing to delete.")
        return

    token = config["HF_TOKEN"]
    if not token:
        print("[cleanup] HF_TOKEN not set — skipping cleanup.")
        return

    backend = (storage or {}).get("backend") \
        or str(config.get("HF_STORAGE_BACKEND", "bucket")).lower()

    if backend == "bucket":
        _cleanup_bucket(config, storage, uploaded_paths, token)
    else:
        _cleanup_repo(config, storage, uploaded_paths, token)


def _cleanup_bucket(config, storage, uploaded_paths, token) -> None:
    from huggingface_hub import batch_bucket_files, list_bucket_tree

    bucket_id = (storage or {}).get("id") or config["HF_BUCKET_ID"]
    print(f"[cleanup] Deleting {len(uploaded_paths)} object(s) from "
          f"bucket:{bucket_id} ...")
    try:
        batch_bucket_files(bucket_id, delete=list(uploaded_paths), token=token)
        for p in uploaded_paths:
            print(f"  ✓ deleted {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ bucket delete failed: {exc}")
        return

    # Verify the objects are actually gone.
    try:
        remaining = {item.path for item in
                     list_bucket_tree(bucket_id, token=token)}
        still = [p for p in uploaded_paths if p in remaining]
        if still:
            print(f"[cleanup] WARNING — still present after delete: {still}")
        else:
            print(f"[cleanup] Verified: all artifacts removed from "
                  f"bucket:{bucket_id}.")
    except Exception as exc:  # noqa: BLE001
        print(f"[cleanup] Could not verify deletion: {exc}")


def _cleanup_repo(config, storage, uploaded_paths, token) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = (storage or {}).get("id") or config["HF_REPO_ID"]
    repo_type = (storage or {}).get("repo_type") or config["HF_REPO_TYPE"]

    print(f"[cleanup] Deleting {len(uploaded_paths)} file(s) from "
          f"{repo_type}:{repo_id} ...")
    for repo_path in uploaded_paths:
        try:
            api.delete_file(
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"[test cleanup] Remove {repo_path}",
            )
            print(f"  ✓ deleted {repo_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ failed to delete {repo_path}: {exc}")

    try:
        remaining = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
        still = [p for p in uploaded_paths if p in remaining]
        if still:
            print(f"[cleanup] WARNING — still present after delete: {still}")
        else:
            print(f"[cleanup] Verified: all artifacts removed from "
                  f"{repo_type}:{repo_id}.")
    except Exception as exc:  # noqa: BLE001
        print(f"[cleanup] Could not verify deletion: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Build drivers
# ─────────────────────────────────────────────────────────────────────────────
def run_via_http(app, host: str, port: int, source_path: str,
                 upload: bool) -> dict:
    """Spin up Flask and call POST /api/knowledge-graph/build over HTTP."""
    import requests

    base_url = f"http://{host}:{port}"
    server = FlaskServerThread(app, host, port)
    server.start()
    try:
        _wait_for_health(base_url)

        print(f"[build] POST {base_url}/api/knowledge-graph/build")
        print(f"[build]   source_path = {source_path}")
        print(f"[build]   upload      = {upload}")
        resp = requests.post(
            f"{base_url}/api/knowledge-graph/build",
            json={"source_path": source_path, "upload": upload},
            timeout=60 * 60,  # KG builds can take a while
        )
        print(f"[build] HTTP {resp.status_code}")
        try:
            result = resp.json()
        except Exception:
            raise RuntimeError(f"Non-JSON response: {resp.text[:500]}")

        print("[build] response:")
        print(json.dumps(result, indent=2)[:4000])

        if resp.status_code != 200 or result.get("status") != "ok":
            raise RuntimeError(f"Build failed: {result}")
        return result
    finally:
        print("[server] shutting down ...")
        server.shutdown()


def run_direct(config, source_path: str, upload: bool) -> dict:
    """Call the service directly — no HTTP, easiest to step through."""
    from api.services.knowledge_graph_service import KnowledgeGraphService

    print("[build] calling KnowledgeGraphService.build() directly")
    print(f"[build]   source_path = {source_path}")
    print(f"[build]   upload      = {upload}")
    service = KnowledgeGraphService(config)
    result = service.build(source_path=source_path, upload=upload)
    result = {"status": "ok", **result}
    print("[build] result:")
    print(json.dumps(result, indent=2)[:4000])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Knowledge Graph from a local folder, upload it to "
                    "HuggingFace storage, then delete it again.",
    )
    parser.add_argument(
        "source_path",
        help="Local folder (or file) to ingest into the knowledge graph.",
    )
    parser.add_argument(
        "--no-upload",
        dest="upload",
        action="store_false",
        help="Build artifacts locally only; do not upload to HuggingFace.",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="Keep the uploaded artifacts on HuggingFace (skip deletion).",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Call the service directly instead of through the Flask HTTP API.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host for the in-process Flask server (default: config HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the in-process Flask server (default: config PORT).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    source = Path(args.source_path).expanduser()
    if not source.exists():
        print(f"[error] source_path does not exist: {source}")
        return 2

    # Build the Flask app + grab its resolved config.
    from app import create_app

    app = create_app()
    config = app.config
    host = args.host or config["HOST"]
    port = args.port or config["PORT"]

    if args.upload and not config.get("HF_TOKEN"):
        print("[warn] HF_TOKEN is not set in backend/.env — upload will fail. "
              "Use --no-upload to build locally only.")

    result: dict
    try:
        if args.direct:
            result = run_direct(config, str(source), args.upload)
        else:
            result = run_via_http(app, host, port, str(source), args.upload)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return 1

    uploaded = result.get("uploaded", []) if isinstance(result, dict) else []
    storage = result.get("storage") if isinstance(result, dict) else None

    if args.upload:
        report_uploaded(storage, uploaded)

    if args.upload and args.cleanup:
        cleanup_hf_artifacts(config, storage, uploaded)
    elif args.upload and not args.cleanup:
        print(f"[cleanup] --no-cleanup set; left {len(uploaded)} file(s) on "
              "HuggingFace.")

    print("[done] pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
