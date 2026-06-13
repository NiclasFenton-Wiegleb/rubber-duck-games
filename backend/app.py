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
    )
