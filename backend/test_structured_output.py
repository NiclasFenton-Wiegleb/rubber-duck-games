from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.services.structured_output import (
    normalize_structured_answer,
    render_duck_questions,
    render_fix_options,
    render_repo_findings,
)


class StructuredOutputTests(unittest.TestCase):
    def test_valid_structured_output_parses_and_preserves_sections(self):
        raw = json.dumps(
            {
                "schema_version": "1.0",
                "session": {
                    "mode": "duck_question",
                    "repo": {
                        "url": "https://example.com/game.git",
                        "branch": "main",
                        "local_path": "/tmp/game",
                    },
                    "user_problem": "Player does not move.",
                },
                "conversation": {
                    "messages": [
                        {
                            "id": "q1",
                            "role": "duck",
                            "kind": "question",
                            "content": "Which input action should move the player?",
                            "intent": "clarify_input",
                            "expects_user_reply": True,
                        }
                    ],
                    "next_prompt_hint": "Check the input map.",
                },
                "repo_findings": [
                    {
                        "id": "finding_1",
                        "title": "Input action may be missing",
                        "summary": "Movement depends on a named input action.",
                        "evidence": [
                            {
                                "file": "player.gd",
                                "symbol": "_physics_process",
                                "reason": "Reads ui_right every frame.",
                            }
                        ],
                        "confidence": "high",
                        "learning_opportunity": {
                            "concept": "Godot input map",
                            "why_it_matters": "Actions decouple keys from code.",
                            "beginner_explanation": "The code asks Godot for an action, not a key.",
                            "suggested_next_step": "Open Project Settings > Input Map.",
                        },
                    }
                ],
                "fix_options": [
                    {
                        "id": "fix_1",
                        "area": "input",
                        "title": "Add missing action",
                        "description": "Create the input action used by the player script.",
                        "complexity": "low",
                        "risk": "low",
                        "recommended": True,
                        "steps": ["Add ui_right", "Test movement"],
                        "tradeoffs": ["Smallest change"],
                    }
                ],
                "refactor_suggestion": {
                    "title": "Name custom actions",
                    "reason": "Project-specific names read better.",
                    "when_to_do_it": "later",
                    "scope": "Input map and player script only.",
                },
            }
        )

        structured, error = normalize_structured_answer(raw)

        self.assertIsNone(error)
        self.assertEqual(structured["conversation"]["messages"][0]["content"], "Which input action should move the player?")
        self.assertEqual(structured["repo_findings"][0]["learning_opportunity"]["concept"], "Godot input map")
        self.assertEqual(structured["fix_options"][0]["complexity"], "low")

    def test_malformed_model_output_returns_friendly_conversation_fallback(self):
        structured, error = normalize_structured_answer("not json")

        self.assertIsNotNone(error)
        self.assertEqual(structured["conversation"]["messages"][0]["kind"], "error")
        self.assertIn("could not shape", structured["conversation"]["messages"][0]["content"])

    def test_duck_questions_render_from_conversation_messages(self):
        structured, _ = normalize_structured_answer(
            json.dumps(
                {
                    "conversation": {
                        "messages": [
                            {
                                "id": "q1",
                                "role": "duck",
                                "kind": "question",
                                "content": "What happens when you press jump?",
                                "expects_user_reply": True,
                            }
                        ],
                        "next_prompt_hint": "Try one input at a time.",
                    }
                }
            )
        )

        html = render_duck_questions(structured)

        self.assertIn("What happens when you press jump?", html)
        self.assertEqual(structured["conversation"]["next_prompt_hint"], "Try one input at a time.")
        self.assertNotIn("Try one input at a time.", html)
        self.assertNotIn("Next", html)

    def test_duck_questions_are_capped_at_one(self):
        structured, _ = normalize_structured_answer(
            json.dumps(
                {
                    "conversation": {
                        "messages": [
                            {"id": "q1", "role": "duck", "kind": "question", "content": "Question one?"},
                            {"id": "q2", "role": "duck", "kind": "question", "content": "Question two?"},
                            {"id": "q3", "role": "duck", "kind": "question", "content": "Question three?"},
                        ],
                    }
                }
            )
        )

        messages = structured["conversation"]["messages"]
        self.assertEqual([message["content"] for message in messages], ["Question one?"])

    def test_repo_findings_render_learning_opportunity(self):
        structured, _ = normalize_structured_answer(
            json.dumps(
                {
                    "repo_findings": [
                        {
                            "title": "Collision layer mismatch",
                            "summary": "Player and floor may not collide.",
                            "learning_opportunity": {
                                "concept": "Collision layers",
                                "why_it_matters": "Layers decide what can touch.",
                                "beginner_explanation": "Both bodies need compatible masks.",
                                "suggested_next_step": "Inspect the player and floor masks.",
                            },
                        }
                    ]
                }
            )
        )

        html = render_repo_findings(structured)

        self.assertIn("Collision layers", html)
        self.assertIn("Layers decide what can touch.", html)
        self.assertIn("Inspect the player and floor masks.", html)

    def test_fix_options_render_complexity_risk_and_recommendation(self):
        structured, _ = normalize_structured_answer(
            json.dumps(
                {
                    "fix_options": [
                        {
                            "area": "input",
                            "title": "Fix input map",
                            "description": "Add the missing action.",
                            "complexity": "low",
                            "risk": "low",
                            "recommended": True,
                            "steps": ["Add action"],
                            "tradeoffs": ["Does not refactor movement"],
                        },
                        {
                            "area": "movement",
                            "title": "Rewrite movement",
                            "description": "Replace the movement function.",
                            "complexity": "high",
                            "risk": "high",
                            "recommended": False,
                        },
                    ]
                }
            )
        )

        html = render_fix_options(structured, requested_area="input")

        self.assertIn("complexity: low", html)
        self.assertIn("risk: low", html)
        self.assertIn("recommended", html)
        self.assertIn("Fix input map", html)
        self.assertNotIn("Rewrite movement", html)


if __name__ == "__main__":
    unittest.main()
