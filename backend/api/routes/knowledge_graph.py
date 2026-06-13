"""
Knowledge Graph API.

POST /api/knowledge-graph/build
    Process and format all folders/files at a designated filepath into a
    Knowledge Graph (chunks + NetworkX graph + FAISS index) and save it
    locally under KG_OUTPUT_DIR/<kg_name>/. Upload to HuggingFace storage
    is opt-in (default: upload=false).

    After a successful build the retrieval service cache is refreshed so
    the new Knowledge Graph is immediately available for RAG queries.

    Request JSON (all optional — falls back to config defaults):
        {
            "source_path": "C:/path/to/folder",   # folder to ingest
            "kg_name": "my-repo-kg",               # subfolder name under artifacts/
            "upload": false                         # push artifacts to HF (opt-in)
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
    kg_name = payload.get("kg_name") or current_app.config.get("KG_NAME", "default")
    upload = payload.get("upload", False)

    service = KnowledgeGraphService(current_app.config)

    try:
        result = service.build(source_path=source_path, kg_name=kg_name, upload=upload)
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
        current_app.logger.exception("Knowledge graph build failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

    # Auto-refresh the retrieval service so the new KG is available immediately
    # for subsequent RAG queries without needing a restart.
    try:
        from api.services.retrieval_service import get_retrieval_service
        retrieval = get_retrieval_service(current_app.config)
        retrieval.refresh()
        current_app.logger.info(
            "Retrieval service refreshed after KG build. Available KGs: %s",
            retrieval.kg_names,
        )
    except Exception as exc:
        current_app.logger.warning(
            "Failed to refresh retrieval service after KG build: %s", exc
        )

    return jsonify({"status": "ok", **result})
