# Rubber Duck Games — Backend

A local Flask backend that exposes three APIs to the frontend:

| Method & Path | Purpose |
| --- | --- |
| `POST /api/knowledge-graph/build` | Process all folders/files at a designated filepath into a Knowledge Graph (chunks + NetworkX graph + FAISS index) and upload the artifacts to a HuggingFace Space storage bucket. |
| `POST /api/query` | Run a user query through the locally hosted small language model for a **simple**, fast answer. |
| `POST /api/query/complex` | Run a user query through the **same** model with extended reasoning + optional Knowledge-Graph/RAG context for a **more complex** output. |

A `GET /health` endpoint is also provided for liveness checks.

## Project structure

```
backend/
├── app.py                  # Flask app factory + entry point
├── config.py               # Env-driven configuration
├── requirements.txt
├── .env.example            # Copy to .env and fill in
├── api/
│   ├── routes/
│   │   ├── knowledge_graph.py   # /api/knowledge-graph/build
│   │   └── query.py             # /api/query  +  /api/query/complex
│   └── services/
│       ├── knowledge_graph_service.py   # ingest → chunk → graph → FAISS → upload
│       └── llm_service.py               # loads SmolLM3-3B, simple + complex generation
└── artifacts/              # (generated) local KG output before upload
```

## Setup

```bash
cd backend

# 1. Create + activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell on Windows

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure environment
copy .env.example .env            # then edit .env (set HF_TOKEN, etc.)
```

## Run

```bash
python app.py
# or
flask --app app run --debug --port 5000
```

The server starts at `http://127.0.0.1:5000`.

## API examples

### Build the Knowledge Graph

```bash
curl -X POST http://127.0.0.1:5000/api/knowledge-graph/build ^
  -H "Content-Type: application/json" ^
  -d "{\"source_path\": \"../test-repo\", \"upload\": true}"
```

Response:

```json
{
  "status": "ok",
  "stats": { "chunks": 1234, "nodes": 5678, "edges": 9012, "node_types": { "...": 0 } },
  "artifacts": { "graph": "...", "faiss_index": "...", "...": "..." },
  "uploaded": ["data/manifest.json", "data/kg/graph.pkl", "..."]
}
```

> Uploading requires a valid `HF_TOKEN` in `.env`. Pass `"upload": false` to build artifacts locally only.

### Simple query

```bash
curl -X POST http://127.0.0.1:5000/api/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"How do I move a CharacterBody2D in Godot?\"}"
```

### Complex query (reasoning + RAG context)

```bash
curl -X POST http://127.0.0.1:5000/api/query/complex ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Design a state machine for a platformer player.\", \"use_context\": true}"
```

## Notes

- **Models load lazily.** The small language model (`niclasfw/smollm3-3b-codex`)
  is only downloaded/loaded on the first `/api/query*` call. Set
  `LLM_LAZY_LOAD=false` to load it at startup instead.
- **GPU:** if you have CUDA, swap `faiss-cpu` for `faiss-gpu` in
  `requirements.txt` and set `LLM_DEVICE=cuda`.
- The KG pipeline mirrors `RubberDuckGames.ipynb`, generalised to ingest any
  folder of text/code files (not just Godot HTML docs).
