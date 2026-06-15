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
tags:
  - build-small-hackathon
  - game-dev
  - debugging
  - small-models
  - off-the-grid
  - off-brand
  - well-tuned
  - field-notes
---

# 🦆 Game Dev Rubber Duck

A friendly AI-powered rubber duck debugger built for hobby game developers. Give it a game repo and a bug report; it clones the project, builds a knowledge graph over the copied source, and asks one focused debugging question grounded in the actual code.

Built by Liam Curran and Niclas FW for the Build Small Hackathon.

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
   model (onto the GPU). Flask runs in the _same_ process, so these singletons are
   shared with the API handlers.
4. **Serves a Gradio UI** — clone a repo, build a project knowledge graph, and
   ask DuckChat for a focused debugging question grounded in that copied code.

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

- Gradio UI: http://127.0.0.1:7860
- Flask API: http://127.0.0.1:5000 (health: `GET /health`)

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

| Variable             | Purpose                                        | Default                             |
| -------------------- | ---------------------------------------------- | ----------------------------------- |
| `KG_STORAGE_DIR`     | Root holding the KG subfolders                 | `/data` (or `backend/artifacts`)    |
| `LLM_DEVICE`         | `cuda` / `cpu` / `auto`                        | auto-detected                       |
| `LLM_LAZY_LOAD`      | Load model on first request instead of startup | `false` in the Space                |
| `RETRIEVAL_TOP_K`    | Chunks kept in the merged context              | `4`                                 |
| `SYSTEM_PROMPT_PATH` | System prompt text file                        | `backend/prompts/system_prompt.txt` |

## Demo Video

- Access the video [here](https://drive.google.com/file/d/1OX5-tTXuQBmyLKD0GtjE7jndCa5uXplD/view?usp=sharing)

## Hackathon Post

- Read the post [here](TODO_ADD_POST_LINK)

## Hackathon Badges

We are submitting for the following Build Small extra-point badges:

- **Off the Grid** — the app uses a small, focused model instead of relying on a huge general-purpose assistant.
- **Off-Brand** — the product is playful and domain-specific: a game-dev rubber duck rather than a generic coding chatbot.
- **Well-Tuned** — DuckChat returns one constrained, structured question at a time so the experience stays fast and focused.
- **Field Notes** — the app includes debug JSON, retrieval sources, and project context so users can see how the duck reached its question.

## What it does

- **Clones a game repo** — bring in a Godot, Unity, pygame, or other hobby game project without pasting snippets by hand.
- **Builds project context** — chunks source files, builds a knowledge graph, and retrieves relevant files/functions for the bug report.
- **Asks one useful question** — DuckChat keeps the debugging loop small by returning a single focused question instead of a wall of fixes.
- **Shows its work** — the raw JSON panel includes retrieval status and sources so users can tell when the answer is grounded in repo context.

## Who it's for

Hobby game developers of any skill level who want to get unstuck without losing
the learning experience. It is especially useful for beginners who have a whole
project, not a perfectly isolated snippet, and need help deciding what to inspect
first.

## Tips for best results

- Use a public repo or fork that the Space can clone.
- Describe what failed, what you expected, and any function, scene, script, or error name you saw.
- Rebuild the knowledge graph after changing branches or pushing a new demo bug.
