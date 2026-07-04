#!/usr/bin/env python3
"""
indexer/indexer.py — CLI entry point.

Walks VAULT_PATH/{quotations,facts,thoughts}/, embeds each .md file
with Ollama, upserts to Qdrant. Idempotent: re-running upserts over
the same point IDs (no duplicates).

Usage:
    python -m indexer.indexer

The script reads VAULT_PATH / QDRANT_URL / OLLAMA_HOST / etc. from
the environment (loaded from .env if present). No CLI args in v0 —
keep the surface small and obvious.

Exit codes (per IMPLEMENTATION_DESIGN.md §"Ingest pipeline"):
    0  success (including empty vault)
    1  Vault path not found
    2  Qdrant unreachable
    3  Embedding model unavailable
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load .env before importing the app modules so config.load() sees
# the values if the app module is imported transitively.
load_dotenv()

from search import config as config_mod  # noqa: E402
from search.ollama_client import (  # noqa: E402
    OllamaClient,
    OllamaError,
    OllamaModelMissing,
    OllamaUnavailable,
)
from search.qdrant_client import QdrantFragments, QdrantUnavailable  # noqa: E402

logger = logging.getLogger(__name__)


# Three canonical fragment-type directories. Other directories at
# the vault root are ignored — v0 only indexes the three types.
FRAGMENT_DIRS: tuple[str, ...] = ("quotations", "facts", "thoughts")


def iter_markdown_files(vault_path: str) -> Iterator[tuple[str, Path]]:
    """
    Walk VAULT_PATH/{quotations,facts,thoughts}/ recursively. Yield
    (type, abs_path) for each .md file. Empty-content files are
    yielded (the caller filters them) so the empty-vault warning
    can distinguish "no files" from "all files were empty".
    """
    root = Path(vault_path)
    for type_dir in FRAGMENT_DIRS:
        type_path = root / type_dir
        if not type_path.is_dir():
            continue
        for path in type_path.rglob("*.md"):
            if path.is_file():
                yield type_dir, path


def _ref_for(vault_path: str, abs_path: Path) -> str:
    """Vault-relative forward-slash path (the Qdrant point id)."""
    return abs_path.relative_to(Path(vault_path)).as_posix()


def make_qdrant_client(cfg: config_mod.Config) -> QdrantClient:
    """Build a qdrant-client from cfg. Caller owns the client."""
    return QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        timeout=30,
    )


def _index_one(
    cfg: config_mod.Config,
    ollama: OllamaClient,
    qdrant: QdrantFragments,
    type_dir: str,
    abs_path: Path,
) -> str:
    """
    Embed and upsert a single fragment. Returns the point id (ref).

    Raises OllamaUnavailable / OllamaModelMissing on embedding
    failure; raises QdrantUnavailable on upsert failure.
    """
    ref = _ref_for(cfg.vault_path, abs_path)
    content = abs_path.read_text(encoding="utf-8", errors="replace")
    if not content.strip():
        # Skip empty files. They were already filtered by the
        # caller, but a race with a deleted file is possible.
        return ""
    vec = ollama.embed_one(content)
    if len(vec) != cfg.embed_dim:
        raise OllamaError(
            f"embedding dim mismatch for {ref}: model returned "
            f"{len(vec)}, EMBED_DIM={cfg.embed_dim}"
        )
    payload = {
        "path": ref,
        "content": content,
        "type": type_dir,
        "filename": abs_path.name,
        "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    qdrant.upsert(ref, vec, payload)
    return ref


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = config_mod.load()

    vault = Path(cfg.vault_path)
    if not vault.is_dir():
        logger.error("vault path not found: %s", cfg.vault_path)
        return 1

    # Walk the vault up-front so we can give an honest "no .md files
    # found" warning instead of just spinning on an empty iterator.
    files = list(iter_markdown_files(cfg.vault_path))
    # Filter empty-content files. The walk doesn't stat each file,
    # so we read once here. For a small vault this is free; for a
    # huge one the indexer is already going to be slow anyway.
    non_empty: list[tuple[str, Path]] = []
    skipped_empty = 0
    for t, p in files:
        try:
            if p.stat().st_size == 0:
                skipped_empty += 1
                continue
        except OSError:
            continue
        non_empty.append((t, p))
    if not files:
        print("warning: no .md files found", file=sys.stderr)
        return 0
    if not non_empty:
        print(
            f"warning: no .md files with non-empty content "
            f"(skipped {skipped_empty} empty files)",
            file=sys.stderr,
        )
        return 0

    # Build clients. The QdrantClient class is patched in tests
    # via monkeypatch on `indexer.indexer.QdrantClient`; the same
    # trick applies to OllamaClient (`indexer.indexer.OllamaClient`).
    try:
        qclient = make_qdrant_client(cfg)
        qdrant = QdrantFragments(
            client=qclient, collection=cfg.qdrant_collection,
        )
        qdrant.ensure_collection(cfg.embed_dim)
    except QdrantUnavailable as e:
        logger.error("qdrant unreachable: %s", e)
        return 2
    except Exception as e:
        # Catch anything else (network errors during client
        # construction, UnexpectedResponse, etc.) and surface as
        # exit code 2 — the design doc says 2 means "qdrant
        # unreachable" and the operator-actionable error here is
        # "Qdrant isn't reachable".
        logger.error("qdrant client construction failed: %s", e)
        return 2

    ollama = OllamaClient(
        host=cfg.ollama_host,
        chat_model=cfg.ollama_chat_model,
        embed_model=cfg.ollama_embed_model,
    )
    # Eager healthcheck on the embed model so we can exit code 3
    # before reading 1000 files we'll never embed.
    if not ollama.healthz():
        logger.error("ollama unreachable at %s", cfg.ollama_host)
        return 3
    try:
        # A single empty-string probe forces Ollama to load the
        # embedding model and surface a 404 if it's missing. Cheaper
        # than embedding all real content only to fail at the end.
        ollama.embed_one("")
    except OllamaModelMissing as e:
        logger.error(
            "embedding model %s not loaded: %s",
            cfg.ollama_embed_model, e,
        )
        return 3
    except OllamaUnavailable as e:
        logger.error("ollama unreachable while loading embed model: %s", e)
        return 3
    except OllamaError as e:
        logger.error("ollama error: %s", e)
        return 3

    total = len(non_empty)
    indexed = 0
    errors = 0
    print(f"indexing {total} files from {cfg.vault_path}")
    for i, (type_dir, abs_path) in enumerate(non_empty, start=1):
        try:
            ref = _index_one(cfg, ollama, qdrant, type_dir, abs_path)
        except OllamaModelMissing as e:
            logger.error("embedding model missing mid-run: %s", e)
            return 3
        except OllamaUnavailable as e:
            logger.error("ollama became unreachable: %s", e)
            return 3
        except OllamaError as e:
            logger.warning("ollama error on %s: %s", abs_path, e)
            errors += 1
            continue
        except QdrantUnavailable as e:
            logger.error("qdrant unreachable: %s", e)
            return 2
        if ref:
            indexed += 1
        if i % 10 == 0 or i == total:
            print(f"{i} / {total} files", flush=True)
    print(f"Done. Indexed: {indexed}, Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())