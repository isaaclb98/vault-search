"""
search/fragment_resolver.py

Vault filename → content lookup.

The Qdrant payload carries the fragment content (so the chat pipeline
can render previews without re-reading disk). The fragment_resolver
sits one layer below that: it reads the canonical source file from
the Obsidian vault when the operator asks for the full content.

For v0, "full content" is just the file body. The web UI hits
GET /api/fragment/<path:ref> on drop, which calls into here to
resolve a vault-relative path to the absolute file location, reads
it, and returns the text.

Two safeguards:
  1. The resolved path must be under VAULT_PATH. Anything else is a
     path-traversal attempt and is rejected as 400.
  2. Files larger than MAX_BYTES are not slurped wholesale — the
     preview is sufficient for the chat panel. The canvas export
     reads from Qdrant payloads (cached) for the same reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_BYTES: int = 64 * 1024  # 64 KiB upper bound on a single fragment.


@dataclass(frozen=True)
class ResolvedFragment:
    """A vault-relative path that resolved to an on-disk file."""
    ref: str
    abs_path: str
    content: str
    truncated: bool
    type: str


def _infer_type(vault_root: Path, abs_path: Path) -> str:
    """
    Infer the fragment type from the parent directory name relative
    to the vault root. "quotations", "facts", "thoughts" → those
    exact strings. Anything else → "" (unknown / outside the three
    canonical types). The chat pipeline treats empty type as
    "no filter available".
    """
    try:
        rel = abs_path.relative_to(vault_root)
    except ValueError:
        return ""
    parts = rel.parts
    if not parts:
        return ""
    # The first path component is the top-level type dir.
    return parts[0]


def resolve(vault_path: str, ref: str) -> ResolvedFragment | None:
    """
    Resolve a vault-relative path to a ResolvedFragment.

    Returns None if:
      - The file doesn't exist on disk.
      - The resolved path is outside vault_path (security check).
      - The file isn't a regular .md file.

    Raises ValueError on suspicious paths (e.g. absolute paths, ".."
    traversal). The caller (route handler) maps that to a 400.

    A file larger than MAX_BYTES is still returned, but with
    `truncated=True` and content cut to the first MAX_BYTES bytes.
    """
    if not ref:
        return None
    # Refuse absolute paths and ".." traversal up-front. The vault
    # root is the only allowed prefix.
    p = Path(ref)
    if p.is_absolute():
        raise ValueError(f"ref must be vault-relative: {ref!r}")
    if any(part == ".." for part in p.parts):
        raise ValueError(f"ref contains '..': {ref!r}")
    if p.suffix.lower() != ".md":
        return None

    root = Path(vault_path)
    abs_path = (root / p).resolve()
    try:
        abs_path.relative_to(root.resolve())
    except ValueError:
        # Resolved path is outside the vault — traversal attempt.
        raise ValueError(f"ref escapes vault root: {ref!r}")
    if not abs_path.is_file():
        return None

    try:
        raw = abs_path.read_bytes()
    except OSError as e:
        logger.warning("failed to read %s: %s", abs_path, e)
        return None

    truncated = len(raw) > MAX_BYTES
    if truncated:
        raw = raw[:MAX_BYTES]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    return ResolvedFragment(
        ref=ref,
        abs_path=str(abs_path),
        content=text,
        truncated=truncated,
        type=_infer_type(root, abs_path),
    )