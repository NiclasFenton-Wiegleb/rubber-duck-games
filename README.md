---
title: "Game Dev Rubber Duck"
emoji: 🦆
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py

pinned: false
suggested_hardware: t4-small
suggested_storage: small
short_description: A friendly AI rubber-duck debugger for hobby game devs.
---

# 🦆 Game Dev Rubber Duck

A friendly AI-powered rubber duck debugger built for hobby game developers. Instead of just handing you the answer, it helps you think through your problem yourself — asking the right questions, nudging you toward best practices, and cheering you on as you figure it out.

## How the Space runs

`app.py` is the single entry point HuggingFace launches (declared above via
`sdk: gradio` + `app_file: app.py`). On startup it:

1. **Configures the environment** for a GPU Space — selects CUDA, eager-loads the
   model (`LLM_LAZY_LOAD=false`) and points `KG_STORAGE_DIR` at the mounted
   persistent storage (`/data`).
2. **Boots the Flask backend** (`backend/app.py`) in a background thread, exposing
   its local APIs at `http://127.0.0.1:5000` (`/api/query`, `/api/query/complex`,
   `/api/knowledge-graph/build`, `/health`).
3. **Warms up** the system prompt (from `backend/prompts/system_prompt.txt`), the
   knowledge-graph retrieval indexes (read from `/data`), and the small language
   model (onto the GPU). Flask runs in the *same* process, so these singletons are
   shared with the API handlers.
4. **Serves a minimal Gradio UI** — a single text box that submits prompts through
   the "simple request" workstream by calling the local Flask API.

> **Persistent storage must be enabled** on the Space for `/data` to be mounted.
> Each knowledge graph lives in its own subfolder, e.g. `/data/godot-docs/…`,
> `/data/my-project/…`; the retriever merges results across all of them. Without
> storage the app still runs and answers without KG context.

## Run & debug locally

```bash
# from the repo root
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only needed for KG building

python app.py
```

- Gradio UI:   http://127.0.0.1:7860
- Flask API:   http://127.0.0.1:5000  (health: `GET /health`)

The UI shows a live **status line** (backend up? model loaded? device? which KGs
were found?) and a **Raw response (debug)** panel with the full JSON returned by
the API. Use the **Options** accordion to toggle KG context and tune
`max_new_tokens` / `temperature`.

Test the API directly:

```bash
curl -X POST http://127.0.0.1:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I move a CharacterBody2D in Godot?"}'
```

Useful environment overrides (see `backend/config.py`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `KG_STORAGE_DIR` | Root holding the KG subfolders | `/data` (or `backend/artifacts`) |
| `LLM_DEVICE` | `cuda` / `cpu` / `auto` | auto-detected |
| `LLM_LAZY_LOAD` | Load model on first request instead of startup | `false` in the Space |
| `RETRIEVAL_TOP_K` | Chunks kept in the merged context | `4` |
| `SYSTEM_PROMPT_PATH` | System prompt text file | `backend/prompts/system_prompt.txt` |

## What it does

- **Rubber duck debugging** — talk through your code problem and the duck helps you reason through it step by step.
- **Beginner-friendly** — clear, approachable explanations for hobbyists.
- **Teaches as you go** — surfaces relevant concepts and best practices in context.
- **Stays in your corner** — guides you to the "aha!" moment rather than dumping a solution.

## Who it's for

Hobby game developers of any skill level who want to get unstuck without losing the
learning experience, understand *why* something isn't working, and pick up good
habits around code structure, debugging, and problem-solving.

## Tips for best results

- Share the relevant snippet of code, not your whole project.
- Describe what you've already tried, and expected vs. actual behaviour.
- Mention your engine/language (Unity/C#, Godot/GDScript, pygame, etc.).
