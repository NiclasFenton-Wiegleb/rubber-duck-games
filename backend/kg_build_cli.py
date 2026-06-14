"""
Standalone Knowledge-Graph build — runs as its OWN short-lived process.

WHY THIS EXISTS (ZeroGPU CUDA-fork safety)
------------------------------------------
On a HuggingFace ZeroGPU Space the hosting process *forks* each ``@spaces.GPU``
worker. The KG build imports **spaCy/thinc** and **sentence-transformers**,
which probe ``torch.cuda`` (``is_available`` / ``device_count``) while importing
and while constructing a ``SentenceTransformer`` — even when pinned to CPU.

On a GPU-less host one of those probes triggers a real, low-level
``torch._C._cuda_init()`` that caches a *failed* CUDA state at the C level. The
``spaces`` torch-CUDA emulation does **not** intercept that low-level path, and
Python-level cache clearing cannot undo it. Every ``@spaces.GPU`` worker
subsequently forked from the host then inherits the poisoned CUDA state and dies
in ``worker_init`` with::

    RuntimeError: No CUDA GPUs are available

Symptom that pinpoints this: the very first generation works, but every
generation *after* a "Clone Repo & Build KG" fails with the error above.

THE FIX
-------
Run the build in this separate process so every CUDA-probing import stays OUT of
the hosting process — the fork stays clean. The child writes all artifacts to
disk (the storage dir is shared) and prints the build-result JSON on a
sentinel-prefixed line for the parent route to parse.

Usage::

    python backend/kg_build_cli.py --source-path <dir> --kg-name <name> [--upload]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the backend package importable (config.py, api/...), regardless of CWD.
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

# The parent process scans stdout for this exact prefix to recover the result.
RESULT_SENTINEL = "KG_BUILD_RESULT_JSON:"


def _config_dict() -> dict:
    """Replicate Flask's ``config.from_object(Config)`` as a plain dict.

    ``KnowledgeGraphService`` only needs a dict-like config (it uses
    ``config[...]`` / ``config.get(...)``), and ``Config`` is fully env-driven,
    so a child process reconstructs an identical configuration from the inherited
    environment.
    """
    from config import Config

    return {k: getattr(Config, k) for k in dir(Config) if k.isupper()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Knowledge Graph in an isolated process (ZeroGPU-safe)."
    )
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--kg-name", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        force=True,
    )

    # Imported here (not at module top) so even a failed import is contained to
    # this throwaway process and never touches the host's CUDA state.
    from api.services.knowledge_graph_service import KnowledgeGraphService

    service = KnowledgeGraphService(_config_dict())
    result = service.build(
        source_path=args.source_path,
        kg_name=args.kg_name,
        upload=args.upload,
    )

    sys.stdout.flush()
    print(RESULT_SENTINEL + json.dumps(result))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
