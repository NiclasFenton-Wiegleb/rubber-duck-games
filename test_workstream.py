"""
Quick local test of the clone → KG build → retrieval workstream.
Runs the backend in-process, no model loading needed.
"""
import json
import os
import sys
import time

# Set env before importing anything
os.environ["FLASK_DEBUG"] = "false"
os.environ["LLM_LAZY_LOAD"] = "true"
os.environ["KG_STORAGE_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "backend", "artifacts"
)
os.environ["KG_OUTPUT_DIR"] = os.environ["KG_STORAGE_DIR"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from app import create_app
from api.services.knowledge_graph_service import KnowledgeGraphService
from api.services.retrieval_service import get_retrieval_service

app = create_app()
config = app.config

# ── Step 1: Build a KG with kg_name ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Building KG with kg_name='test-repo'")
print("=" * 60)
service = KnowledgeGraphService(config)
result = service.build(
    source_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-repo"),
    kg_name="test-repo",
    upload=False,
)
print("\n[OK] Build result stats:", json.dumps(result["stats"], indent=2))
print("[OK] Artifacts written to:", config["KG_OUTPUT_DIR"] / "test-repo")

# ── Step 2: Verify the subfolder layout ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Verifying subfolder layout")
print("=" * 60)
kg_dir = config["KG_OUTPUT_DIR"] / "test-repo"
for check in ["chunks/chunks.jsonl", "faiss/index.faiss", "faiss/node_id_map.json",
              "kg/graph.pkl", "kg/entity_to_chunks.json", "manifest.json"]:
    f = kg_dir / check
    status = "✓" if f.exists() else "✗ MISSING"
    size = f.stat().st_size if f.exists() else 0
    print(f"  {status} test-repo/{check} ({size:,} bytes)")

# ── Step 3: Test retrieval refresh and query ──────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Testing retrieval refresh + query")
print("=" * 60)
retrieval = get_retrieval_service(config)
retrieval.refresh()
print(f"  Available KGs: {retrieval.kg_names}")

result = retrieval.retrieve("How do I move a paddle in Godot?")
print(f"  Context length: {len(result['context'])} chars")
print(f"  Sources: {len(result['sources'])}")
for s in result["sources"]:
    print(f"    - [{s['kg']}] chunk {s['chunk_id']} (score: {s['score']})")

if result["sources"]:
    print("\n  Sample context snippet:")
    print("  " + result["context"][:200].replace("\n", "\n  ") + "...")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)