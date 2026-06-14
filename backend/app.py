"""
Rubber Duck Games — Flask backend entry point.

Exposes local APIs to the frontend:
  - POST /api/knowledge-graph/build  → build a Knowledge Graph from a folder
                                        and upload it to a HuggingFace Space.
  - POST /api/query                  → run a user query through the local SLM
                                        (simple, fast answer).
  - POST /api/query/complex          → run a user query through the same SLM
                                        with extended reasoning + KG/RAG context.

Run locally:
    python app.py
or:
    flask --app app run --debug --port 5000
"""

import logging
from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from api.routes.knowledge_graph import kg_bp
from api.routes.query import query_bp

log = logging.getLogger("backend")


def _detect_and_log_device(config) -> str:
    """
    Detect the best available device for model inference at startup.

    Returns the resolved device string ("cuda", "mps", or "cpu") and logs
    a prominent message so operators can see at a glance where the model
    will run.

    On HuggingFace ZeroGPU the real GPU is only attached inside a
    ``@spaces.GPU`` worker; the host process sees a patched stub. We detect
    ZeroGPU and skip CUDA probing to avoid misleading log messages (and
    potential CUDA-context pollution).
    """
    try:
        import torch
    except ImportError:
        log.info("PyTorch not installed – model inference will use CPU "
                 "once torch becomes available.")
        return "cpu"

    # ZeroGPU: the real GPU is only inside @spaces.GPU workers.
    # The env var LLM_DEVICE is already set to "cuda" by app.py's
    # _configure_environment() so the dtype resolver picks bfloat16.
    # We don't probe torch.cuda here because the spaces-patched probe
    # returns fake data in the host, and probing may contribute to the
    # "CUDA already initialised" fork-worker problem.
    is_zero_gpu = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true", "yes")
    if is_zero_gpu:
        log.info("=" * 60)
        log.info("ZeroGPU runtime — GPU will be attached inside "
                 "@spaces.GPU workers; hosting process uses CPU. "
                 "Model weights will load in bfloat16 inside the "
                 "GPU context on first query.")
        log.info("=" * 60)
        return "cuda"  # LLM_DEVICE=cuda tells the dtype resolver to use bf16

    device = (config.get("LLM_DEVICE") or "auto").lower()

    cuda_available = torch.cuda.is_available()
    mps_available = (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    resolved: str
    if device == "cuda":
        resolved = "cuda" if cuda_available else "cpu"
    elif device == "mps":
        resolved = "mps" if mps_available else "cpu"
    elif device == "cpu":
        resolved = "cpu"
    else:  # "auto" or anything unrecognised
        if cuda_available:
            resolved = "cuda"
        elif mps_available:
            resolved = "mps"
        else:
            resolved = "cpu"

    if resolved == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("=" * 60)
        log.info("GPU DETECTED: %s (%.1f GiB VRAM)", props.name,
                 props.total_memory / (1024 ** 3))
        log.info("Model will be loaded onto CUDA (GPU) with bfloat16 precision.")
        log.info("=" * 60)
    elif resolved == "mps":
        log.info("=" * 60)
        log.info("GPU DETECTED: Apple Metal (MPS)")
        log.info("Model will be loaded onto MPS with float16 precision.")
        log.info("=" * 60)
    else:
        log.info("=" * 60)
        log.info("NO GPU DETECTED — model will run on CPU (float32).")
        log.info("=" * 60)

    return resolved


def create_app(config_class: type = Config) -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configure Flask's logger to output request details for all HTTP methods.
    app.logger.setLevel(logging.INFO)
    if not app.logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
        ))
        app.logger.addHandler(handler)
    # Also set the werkzeug logger level so POST requests are logged.
    logging.getLogger("werkzeug").setLevel(logging.INFO)

    # ── GPU detection & early model loading ───────────────────────────────
    _detect_and_log_device(app.config)

    if not app.config.get("LLM_LAZY_LOAD", True):
        log.info("LLM_LAZY_LOAD is false — pre-loading model at startup ...")
        try:
            from api.services.llm_service import get_llm_service
            llm = get_llm_service(app.config)
            log.info("Model pre-loaded successfully on device '%s'.", llm.device)
        except Exception as exc:
            log.error("Failed to pre-load model at startup: %s", exc)

    # Allow the local frontend (e.g. Vite/React dev server) to call these APIs.
    CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    # Register API blueprints.
    app.register_blueprint(kg_bp, url_prefix="/api/knowledge-graph")
    app.register_blueprint(query_bp, url_prefix="/api")

    @app.get("/health")
    def health():
        """Simple liveness probe."""
        return jsonify({"status": "ok", "service": "rubber-duck-games-backend"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config["DEBUG"],
        use_reloader=False,
    )
