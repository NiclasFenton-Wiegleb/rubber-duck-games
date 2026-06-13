"""
Query APIs — run a user query through the locally hosted small language model.

POST /api/query
    Simple, fast answer (the "simple request" workstream):
      1. system prompt is loaded from a local text file,
      2. the most relevant context across all Knowledge Graphs in the attached
         storage bucket is retrieved and added to the prompt,
      3. the assembled prompt is run straight through the SLM on the local GPU,
      4. the formatted answer + sources are returned.

    Request JSON:
        { "query": "How do I move a KinematicBody2D?",
          "use_context": true,         # optional, pull KG context (default true)
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
          "context_used": true|false,
          "sources": [ { "kg": ..., "chunk_id": ..., "score": ... }, ... ],
          "mode": "simple" | "complex" }
"""

import logging
import time

from flask import Blueprint, current_app, jsonify, request

from api.services.llm_service import get_llm_service

log = logging.getLogger("query_routes")

query_bp = Blueprint("query", __name__)


def _extract_query():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    return (query or None), payload


@query_bp.before_request
def _log_request():
    log.info("--> %s %s (content_type=%s, content_length=%d)",
             request.method, request.path,
             request.content_type, request.content_length or 0)


@query_bp.after_request
def _log_response(response):
    elapsed = getattr(request, "_start_time", None)
    if elapsed is not None:
        elapsed = time.time() - elapsed
        log.info("<-- %s %s → %d (%.3fs)",
                 request.method, request.path,
                 response.status_code, elapsed)
    else:
        log.info("<-- %s %s → %d",
                 request.method, request.path,
                 response.status_code)
    return response


@query_bp.post("/query")
def query_simple():
    request._start_time = time.time()  # type: ignore[attr-defined]
    query, payload = _extract_query()
    if not query:
        return jsonify({"status": "error", "message": "Missing 'query'."}), 400

    log.info("Simple query: '%s' (use_context=%s, max_new_tokens=%s, temp=%s)",
             query[:80], payload.get("use_context", True),
             payload.get("max_new_tokens"), payload.get("temperature"))

    llm = get_llm_service(current_app.config)
    try:
        result = llm.generate_simple(
            query,
            max_new_tokens=payload.get("max_new_tokens"),
            temperature=payload.get("temperature"),
            use_context=payload.get("use_context", True),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Simple query failed: %s", exc)
        current_app.logger.exception("Simple query failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

    log.info("Simple query succeeded: answer_len=%d, sources=%d",
             len(result.get("answer", "")), len(result.get("sources", [])))
    return jsonify({"status": "ok", "mode": "simple", "query": query, **result})


@query_bp.post("/query/complex")
def query_complex():
    request._start_time = time.time()  # type: ignore[attr-defined]
    query, payload = _extract_query()
    if not query:
        return jsonify({"status": "error", "message": "Missing 'query'."}), 400

    log.info("Complex query: '%s' (use_context=%s, max_new_tokens=%s, temp=%s)",
             query[:80], payload.get("use_context", True),
             payload.get("max_new_tokens"), payload.get("temperature"))

    llm = get_llm_service(current_app.config)
    try:
        result = llm.generate_complex(
            query,
            use_context=payload.get("use_context", True),
            max_new_tokens=payload.get("max_new_tokens"),
            temperature=payload.get("temperature"),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Complex query failed: %s", exc)
        current_app.logger.exception("Complex query failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

    log.info("Complex query succeeded: answer_len=%d, reasoning_len=%d, sources=%d",
             len(result.get("answer", "")),
             len(result.get("reasoning") or ""),
             len(result.get("sources", [])))
    return jsonify({"status": "ok", "mode": "complex", "query": query, **result})
