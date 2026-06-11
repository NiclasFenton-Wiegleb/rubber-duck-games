"""Structured duck response parsing, defaults, and HTML rendering."""

from __future__ import annotations

import html
import json
from copy import deepcopy
from json import JSONDecodeError
from typing import Any

LEVELS = {"low", "medium", "high"}
REFACTOR_TIMING = {"now", "after_fix", "later"}


def default_structured_answer(
    repo_url: str = "",
    branch: str = "main",
    local_path: str = "",
    user_problem: str = "",
) -> dict[str, Any]:
    """Return the minimum structured response the UI can safely render."""
    return {
        "schema_version": "1.0",
        "session": {
            "mode": "duck_question",
            "repo": {
                "url": repo_url or "",
                "branch": branch or "main",
                "local_path": local_path or "",
            },
            "user_problem": user_problem or "",
        },
        "conversation": {
            "messages": [],
            "next_prompt_hint": "Tell the duck what happened when you tried the next check.",
        },
        "repo_findings": [],
        "fix_options": [],
        "refactor_suggestion": {
            "title": "Keep the first fix small",
            "reason": "The repo needs one confirmed failing behavior before a larger cleanup is useful.",
            "when_to_do_it": "after_fix",
            "scope": "Return to this once the smallest working fix is tested.",
        },
    }


def normalize_structured_answer(
    raw_answer: str,
    repo_url: str = "",
    branch: str = "main",
    local_path: str = "",
    user_problem: str = "",
) -> tuple[dict[str, Any], str | None]:
    """Parse model JSON, apply defaults, and return (structured, error)."""
    base = default_structured_answer(repo_url, branch, local_path, user_problem)
    try:
        parsed = _extract_json_object(raw_answer)
    except (TypeError, ValueError, JSONDecodeError) as exc:
        fallback = deepcopy(base)
        fallback["conversation"]["messages"] = [
            {
                "id": "duck_parse_error",
                "role": "duck",
                "kind": "error",
                "content": (
                    "I could not shape that answer into the duck session format. "
                    "Try asking again with a little more detail about the broken behavior."
                ),
                "intent": "recover_from_format_error",
                "expects_user_reply": True,
            }
        ]
        return fallback, str(exc)

    if not isinstance(parsed, dict):
        return base, "Structured response was not a JSON object."

    structured = _merge_defaults(base, parsed)
    _normalize_session(structured, base)
    _normalize_conversation(structured)
    _normalize_repo_findings(structured)
    _normalize_fix_options(structured)
    _normalize_refactor(structured)
    return structured, None


def _extract_json_object(raw_answer: str) -> dict[str, Any]:
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise ValueError("Empty model response.")

    text = raw_answer.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    first_brace = text.find("{")
    if first_brace < 0:
        raise ValueError("No JSON object found in model response.")
    parsed, _ = decoder.raw_decode(text[first_brace:])
    return parsed


def _merge_defaults(base: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in parsed.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip() or default


def _clean_level(value: Any, default: str = "medium") -> str:
    text = _clean_text(value, default).lower()
    return text if text in LEVELS else default


def _clean_timing(value: Any, default: str = "after_fix") -> str:
    text = _clean_text(value, default).lower()
    return text if text in REFACTOR_TIMING else default


def _clean_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return default


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _normalize_session(structured: dict[str, Any], base: dict[str, Any]) -> None:
    session = structured["session"] if isinstance(structured.get("session"), dict) else {}
    repo = session.get("repo") if isinstance(session.get("repo"), dict) else {}
    structured["session"] = {
        "mode": _clean_text(session.get("mode"), "duck_question"),
        "repo": {
            "url": _clean_text(repo.get("url"), base["session"]["repo"]["url"]),
            "branch": _clean_text(repo.get("branch"), base["session"]["repo"]["branch"]),
            "local_path": _clean_text(repo.get("local_path"), base["session"]["repo"]["local_path"]),
        },
        "user_problem": _clean_text(session.get("user_problem"), base["session"]["user_problem"]),
    }


def _normalize_conversation(structured: dict[str, Any]) -> None:
    conversation = structured.get("conversation") if isinstance(structured.get("conversation"), dict) else {}
    messages = []
    raw_messages = conversation.get("messages") if isinstance(conversation.get("messages"), list) else []
    for index, raw in enumerate(raw_messages, start=1):
        if not isinstance(raw, dict):
            continue
        content = _clean_text(raw.get("content"))
        if not content:
            continue
        messages.append(
            {
                "id": _clean_text(raw.get("id"), f"duck_message_{index}"),
                "role": _clean_text(raw.get("role"), "duck"),
                "kind": _clean_text(raw.get("kind"), "question"),
                "content": content,
                "intent": _clean_text(raw.get("intent"), "clarify_symptom"),
                "expects_user_reply": _clean_bool(raw.get("expects_user_reply"), True),
            }
        )
    structured["conversation"] = {
        "messages": messages,
        "next_prompt_hint": _clean_text(
            conversation.get("next_prompt_hint"),
            "Tell the duck what you tried next.",
        ),
    }


def _normalize_repo_findings(structured: dict[str, Any]) -> None:
    findings = []
    raw_findings = structured.get("repo_findings") if isinstance(structured.get("repo_findings"), list) else []
    for index, raw in enumerate(raw_findings, start=1):
        if not isinstance(raw, dict):
            continue
        title = _clean_text(raw.get("title"), f"Finding {index}")
        evidence = []
        raw_evidence = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "file": _clean_text(item.get("file")),
                    "symbol": _clean_text(item.get("symbol")),
                    "reason": _clean_text(item.get("reason")),
                }
            )
        learning = raw.get("learning_opportunity") if isinstance(raw.get("learning_opportunity"), dict) else {}
        findings.append(
            {
                "id": _clean_text(raw.get("id"), f"finding_{index}"),
                "title": title,
                "summary": _clean_text(raw.get("summary")),
                "evidence": evidence,
                "confidence": _clean_level(raw.get("confidence")),
                "learning_opportunity": {
                    "concept": _clean_text(learning.get("concept"), title),
                    "why_it_matters": _clean_text(learning.get("why_it_matters")),
                    "beginner_explanation": _clean_text(learning.get("beginner_explanation")),
                    "suggested_next_step": _clean_text(learning.get("suggested_next_step")),
                },
            }
        )
    structured["repo_findings"] = findings


def _normalize_fix_options(structured: dict[str, Any]) -> None:
    options = []
    raw_options = structured.get("fix_options") if isinstance(structured.get("fix_options"), list) else []
    for index, raw in enumerate(raw_options, start=1):
        if not isinstance(raw, dict):
            continue
        title = _clean_text(raw.get("title"), f"Fix option {index}")
        options.append(
            {
                "id": _clean_text(raw.get("id"), f"fix_{index}"),
                "area": _clean_text(raw.get("area"), "general"),
                "title": title,
                "description": _clean_text(raw.get("description")),
                "complexity": _clean_level(raw.get("complexity")),
                "risk": _clean_level(raw.get("risk")),
                "recommended": _clean_bool(raw.get("recommended"), index == 1),
                "steps": _clean_string_list(raw.get("steps")),
                "tradeoffs": _clean_string_list(raw.get("tradeoffs")),
            }
        )
    if options and not any(option["recommended"] for option in options):
        options[0]["recommended"] = True
    structured["fix_options"] = options


def _normalize_refactor(structured: dict[str, Any]) -> None:
    raw = structured.get("refactor_suggestion") if isinstance(structured.get("refactor_suggestion"), dict) else {}
    structured["refactor_suggestion"] = {
        "title": _clean_text(raw.get("title"), "Keep the first fix small"),
        "reason": _clean_text(raw.get("reason")),
        "when_to_do_it": _clean_timing(raw.get("when_to_do_it")),
        "scope": _clean_text(raw.get("scope")),
    }


def render_duck_questions(structured: dict[str, Any]) -> str:
    messages = structured.get("conversation", {}).get("messages", [])
    if not messages:
        messages = [
            {
                "role": "duck",
                "kind": "question",
                "content": "What behavior did you expect, and what happened instead?",
                "intent": "clarify_symptom",
                "expects_user_reply": True,
            }
        ]
    lines = []
    for message in messages:
        role = _clean_text(message.get("role"), "duck")
        avatar = "You" if role == "user" else "D"
        kicker = "You" if role == "user" else f"Duck - {_clean_text(message.get('kind'), 'question')}"
        reply_tag = '<span class="tag tag-green">reply helpful</span>' if message.get("expects_user_reply") else ""
        lines.append(
            '<div class="chat-line">'
            f'<div class="avatar">{html.escape(avatar)}</div>'
            '<div class="chat-card">'
            f'<div class="card-kicker">{html.escape(kicker)}</div>'
            f'<p>{html.escape(_clean_text(message.get("content")))}</p>'
            f'{reply_tag}'
            '</div>'
            '</div>'
        )
    hint = _clean_text(structured.get("conversation", {}).get("next_prompt_hint"))
    if hint:
        lines.append(f'<div class="info-card"><div class="card-kicker">Next</div>{html.escape(hint)}</div>')
    return f'<div class="chat-thread">{"".join(lines)}</div>'


def render_repo_findings(structured: dict[str, Any]) -> str:
    findings = structured.get("repo_findings", [])
    if not findings:
        return (
            '<div class="info-card"><div class="card-kicker">Repo Findings</div>'
            'Ask the duck to inspect the cloned project for file-level findings.'
            '</div>'
        )
    cards = []
    for finding in findings:
        evidence = "".join(
            '<li class="evidence-item" style="color:#243247; opacity:1;">'
            f'<strong class="evidence-file" style="color:#172033; font-weight:800;">'
            f'{html.escape(_clean_text(item.get("file"), "Unknown file"))}</strong>'
            f' <span class="evidence-symbol" style="color:#40516a; font-weight:650;">'
            f'{html.escape(_clean_text(item.get("symbol")))}</span>'
            f' <span class="evidence-reason" style="color:#40516a; font-weight:650;">'
            f'- {html.escape(_clean_text(item.get("reason")))}</span>'
            '</li>'
            for item in finding.get("evidence", [])
        )
        evidence_list = (
            f'<ul class="evidence-list" style="color:#243247; opacity:1;">{evidence}</ul>'
            if evidence else ""
        )
        learning = finding.get("learning_opportunity", {})
        cards.append(
            '<div class="info-card">'
            f'<div class="card-kicker">Repo Finding - {html.escape(_clean_text(finding.get("confidence"), "medium"))}</div>'
            f'<strong>{html.escape(_clean_text(finding.get("title")))}</strong>'
            f'<p>{html.escape(_clean_text(finding.get("summary")))}</p>'
            f'{evidence_list}'
            f'<span class="tag tag-blue">{html.escape(_clean_text(learning.get("concept"), "Learning opportunity"))}</span>'
            f'<p><strong>Why it matters:</strong> {html.escape(_clean_text(learning.get("why_it_matters")))}</p>'
            f'<p><strong>Beginner note:</strong> {html.escape(_clean_text(learning.get("beginner_explanation")))}</p>'
            f'<p><strong>Next step:</strong> {html.escape(_clean_text(learning.get("suggested_next_step")))}</p>'
            '</div>'
        )
    return f'<div class="stack-list">{"".join(cards)}</div>'


def render_fix_options(structured: dict[str, Any], requested_area: str = "") -> str:
    options = structured.get("fix_options", [])
    requested = requested_area.strip().lower()
    if requested:
        filtered = [
            option for option in options
            if requested in " ".join(
                [
                    _clean_text(option.get("area")),
                    _clean_text(option.get("title")),
                    _clean_text(option.get("description")),
                ]
            ).lower()
        ]
        options = filtered or options
    if not options:
        return (
            '<div class="info-card"><div class="card-kicker">Fix Options</div>'
            'Ask the duck to generate focused fix options from the repo context.'
            '</div>'
        )
    cards = []
    for option in options:
        steps = "".join(
            f'<div class="fix-step">'
            f'<span class="fix-step-index">{index}.</span>'
            f'{html.escape(step)}'
            f'</div>'
            for index, step in enumerate(option.get("steps", []), start=1)
        )
        tradeoffs = "".join(
            f'<div class="fix-tradeoff">{html.escape(item)}</div>'
            for item in option.get("tradeoffs", [])
        )
        steps_list = f'<div class="fix-steps">{steps}</div>' if steps else ""
        tradeoffs_list = (
            '<p class="fix-tradeoffs-heading"><strong>Tradeoffs</strong></p>'
            f'<div class="fix-tradeoffs">{tradeoffs}</div>'
            if tradeoffs else ""
        )
        recommended = '<span class="tag tag-green">recommended</span>' if option.get("recommended") else ""
        cards.append(
            '<div class="info-card">'
            f'<div class="card-kicker">Fix Option - {html.escape(_clean_text(option.get("area"), "general"))}</div>'
            f'<strong>{html.escape(_clean_text(option.get("title")))}</strong>'
            f'<p>{html.escape(_clean_text(option.get("description")))}</p>'
            f'{recommended}'
            f'<span class="tag tag-blue">complexity: {html.escape(_clean_text(option.get("complexity"), "medium"))}</span>'
            f'<span class="tag tag-gold">risk: {html.escape(_clean_text(option.get("risk"), "medium"))}</span>'
            f'{steps_list}'
            f'{tradeoffs_list}'
            '</div>'
        )
    return f'<div class="stack-list">{"".join(cards)}</div>'


def render_refactor(structured: dict[str, Any]) -> str:
    refactor = structured.get("refactor_suggestion", {})
    return (
        '<div class="stack-list"><div class="info-card">'
        f'<div class="card-kicker">Refactor Suggestion - {html.escape(_clean_text(refactor.get("when_to_do_it"), "after_fix"))}</div>'
        f'<strong>{html.escape(_clean_text(refactor.get("title"), "Keep the first fix small"))}</strong>'
        f'<p>{html.escape(_clean_text(refactor.get("reason")))}</p>'
        f'<span class="tag tag-blue">{html.escape(_clean_text(refactor.get("scope"), "small scope"))}</span>'
        '</div></div>'
    )
