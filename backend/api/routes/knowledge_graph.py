"""
Knowledge Graph API.

POST /api/knowledge-graph/build
    Process and format all folders/files at a designated filepath into a
    Knowledge Graph (chunks + NetworkX graph + FAISS index) and upload the
    artifacts to the configured HuggingFace Space storage bucket.

    Request JSON (all optional — falls back to config defaults):
        {
            "source_path": "C:/path/to/folder",   # folder to ingest
            "upload": true                          # push artifacts to HF
        }

    Response JSON:
        {
            "status": "ok",
            "stats": { "chunks": ..., "nodes": ..., "edges": ... },
            "artifacts": { ... local paths ... },
            "uploaded": [ ... repo paths ... ]
        }
"""

from flask import Blueprint, current_app, jsonify, request

from api.services.knowledge_graph_service import KnowledgeGraphService

kg_bp = Blueprint("knowledge_graph", __name__)


@kg_bp.post("/build")
def build_knowledge_graph():
    payload = request.get_json(silent=True) or {}

    source_path = payload.get("source_path") or current_app.config["KG_SOURCE_PATH"]
    upload = payload.get("upload", True)

    service = KnowledgeGraphService(current_app.config)

    try:
        result = service.build(source_path=source_path, upload=upload)
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
        current_app.logger.exception("Knowledge graph build failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

    return jsonify({"status": "ok", **result})
