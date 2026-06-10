"""
Query APIs — run a user query through the locally hosted small language model.

POST /api/query
    Simple, fast answer. Runs the prompt straight through the SLM.

    Request JSON:
        { "query": "How do I move a KinematicBody2D?",
          "max_new_tokens": 512,        # optional
          "temperature": 0.7 }          # optional

POST /api/query/complex
    Same SLM, richer output: enables extended reasoning ("thinking") and can
    augment the prompt with Knowledge-Graph / RAG context for a more detailed,
    structured answer.

    Request JSON:
        { "query": "...",
          "use_context": true,          # optional, pull KG/RAG context
          "max_new_tokens": 2048,       # optional
          "temperature": 0.7 }          # optional

Both return:
        { "status": "ok",
          "query": "...",
          "answer": "...",
          "reasoning": "...",           # complex endpoint only (may be null)
          "mode": "simple" | "complex" }
"""

from flask import Blueprint, current_app, jsonify, request

from api.services.llm_service import get_llm_service

query_bp = Blueprint("query", __name__)


def _extract_query() -> str | None:
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    return query or None, payload


@query_bp.post("/query")
def query_simple():
    query, payload = _extract_query()
    if not query:
        return jsonify({"status": "error", "message": "Missing 'query'."}), 400

    llm = get_llm_service(current_app.config)
    try:
        result = llm.generate_simple(
            query,
            max_new_tokens=payload.get("max_new_tokens"),
            temperature=payload.get("temperature"),
        )
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Simple query failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

    return jsonify({"status": "ok", "mode": "simple", "query": query, **result})


@query_bp.post("/query/complex")
def query_complex():
    query, payload = _extract_query()
    if not query:
        return jsonify({"status": "error", "message": "Missing 'query'."}), 400

    llm = get_llm_service(current_app.config)
    try:
        result = llm.generate_complex(
            query,
            use_context=payload.get("use_context", True),
            max_new_tokens=payload.get("max_new_tokens"),
            temperature=payload.get("temperature"),
        )
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Complex query failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

    return jsonify({"status": "ok", "mode": "complex", "query": query, **result})
