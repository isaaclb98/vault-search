"""
search/canvas.py — canvas CRUD + export.

Storage: one JSON file per canvas in CANVAS_DIR. Filename = slug
derived from canvas name (URL-safe).

Concurrency: single-user. A per-canvas in-process lock (sufficient
for the v0 scope). v1+ may need real filesystem locking.

Atomic writes: save to `<file>.tmp` then rename. A crash mid-write
leaves the .tmp behind and the original file untouched.

Bad-JSON-on-load: surfaces a CanvasCorrupted error. The route
handler maps it to 500 with a "Canvas corrupted" message.

Schema is fixed at module load time — see _validate_payload below.
The CanvasDetail / CanvasSummary models in models.py are the wire
shape; _validate_payload normalises raw dicts to those shapes.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from search.models import (
    CanvasDetailResponse,
    CanvasItem,
    CanvasListResponse,
    CanvasSaveResponse,
    CanvasSummary,
)

logger = logging.getLogger(__name__)

# Filename-safe slug regex. Lowercase letters, digits, hyphens.
_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_SLUG_COLLAPSE_RE = re.compile(r"-{2,}")


class CanvasCorrupted(Exception):
    """A canvas JSON file was found but couldn't be parsed."""


def slugify(name: str) -> str:
    """
    Convert a human canvas name to a URL-safe filename stem.

    Examples:
      "Calasso on Sacrifice" -> "calasso-on-sacrifice"
      "  Foo / Bar  "        -> "foo-bar"
      ""                     -> "" (caller must guard)
    """
    s = (name or "").strip().lower()
    s = s.replace(" ", "-").replace("_", "-")
    s = _SLUG_RE.sub("", s)
    s = _SLUG_COLLAPSE_RE.sub("-", s).strip("-")
    return s


def _now_iso() -> str:
    """ISO 8601 UTC timestamp with 'Z' suffix (not '+00:00')."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> datetime:
    """
    Parse an ISO 8601 string, tolerating either 'Z' or '+00:00'.
    Falls back to now() on garbage so a corrupt file doesn't kill
    the list endpoint — it just gets a 'now' timestamp on the
    summary row.
    """
    if not s:
        return datetime.now(timezone.utc)
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1] + "+00:00")
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)


def _validate_payload(data: dict[str, Any]) -> CanvasDetailResponse:
    """
    Validate a raw dict against the CanvasDetailResponse schema.

    Accepts minor variants (missing fields default; extra fields
    ignored) so we can extend the schema without breaking old saves.
    """
    name = str(data.get("name") or "Untitled")
    slug = str(data.get("slug") or slugify(name))
    created = _parse_iso(str(data.get("created") or ""))
    modified = _parse_iso(str(data.get("modified") or ""))
    raw_items = data.get("items") or []
    if not isinstance(raw_items, list):
        raise CanvasCorrupted(f"items must be a list, got {type(raw_items).__name__}")
    items: list[CanvasItem] = []
    for i, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise CanvasCorrupted(f"item {i} must be an object")
        try:
            items.append(CanvasItem(
                ref=str(raw.get("ref") or ""),
                order=int(raw.get("order") or i),
                x=float(raw.get("x") or 0),
                y=float(raw.get("y") or 0),
                annotation=str(raw.get("annotation") or ""),
            ))
        except (ValueError, TypeError) as e:
            raise CanvasCorrupted(f"item {i}: {e}") from e
    return CanvasDetailResponse(
        slug=slug,
        name=name,
        created=created,
        modified=modified,
        items=items,
    )


class CanvasStore:
    """
    On-disk canvas store. One JSON file per canvas under canvas_dir.

    All public methods are thread-safe (single in-process lock per
    canvas). The locks are held for the duration of read-modify-write
    cycles only — concurrent reads of different canvases don't block
    each other.
    """

    def __init__(self, canvas_dir: str):
        self.canvas_dir = Path(canvas_dir)
        self.canvas_dir.mkdir(parents=True, exist_ok=True)
        # Per-slug lock map. Created on demand, never removed (the
        # OS will reclaim memory when the process exits). Locks are
        # keyed by the absolute target path so two callers asking
        # for the same slug get the same lock.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_meta_lock = threading.Lock()

    # ---------------------- Internal ----------------------

    def _path_for(self, slug: str) -> Path:
        return self.canvas_dir / f"{slug}.json"

    def _lock_for(self, path: Path) -> threading.Lock:
        key = str(path.resolve())
        with self._locks_meta_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _read_raw(self, path: Path) -> dict:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None  # type: ignore[return-value]
        except json.JSONDecodeError as e:
            raise CanvasCorrupted(f"{path.name}: {e}") from e

    def _atomic_write(self, path: Path, data: dict) -> None:
        """
        Write JSON atomically: write to <path>.tmp, fsync, rename.
        On a crash mid-write, the original file is untouched.
        """
        tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            import os as _os
            _os.fsync(f.fileno())
        tmp.replace(path)

    # ---------------------- Public CRUD ----------------------

    def list(self) -> CanvasListResponse:
        """List all canvases sorted by modified (newest first)."""
        summaries: list[CanvasSummary] = []
        for path in self.canvas_dir.glob("*.json"):
            try:
                raw = self._read_raw(path)
            except CanvasCorrupted:
                # Skip corrupted files but keep the directory listing
                # usable. The detail endpoint will surface the error.
                logger.warning("skipping corrupted canvas: %s", path)
                continue
            if raw is None:
                continue
            detail = _validate_payload(raw)
            summaries.append(CanvasSummary(
                slug=detail.slug,
                name=detail.name,
                created=detail.created,
                modified=detail.modified,
                item_count=len(detail.items),
            ))
        summaries.sort(key=lambda s: s.modified, reverse=True)
        return CanvasListResponse(canvases=summaries)

    def get(self, slug: str) -> CanvasDetailResponse:
        """Read a single canvas. Raises CanvasCorrupted on bad JSON."""
        path = self._path_for(slug)
        lock = self._lock_for(path)
        with lock:
            raw = self._read_raw(path)
        if raw is None:
            return None  # type: ignore[return-value]
        return _validate_payload(raw)

    def save(
        self,
        name: str,
        items: list[CanvasItem],
        slug: str | None = None,
    ) -> CanvasSaveResponse:
        """
        Create or update a canvas.

        If `slug` is provided and matches an existing file, the
        existing record is updated (created timestamp preserved,
        modified timestamp refreshed). If `slug` is None or doesn't
        match an existing file, a new canvas is created with a
        fresh slug derived from `name`.

        Empty name or empty slug after slugify raises ValueError.
        """
        effective_slug = (slug or "").strip() or slugify(name)
        if not effective_slug:
            raise ValueError("canvas name produces an empty slug")
        if not name.strip():
            raise ValueError("canvas name cannot be empty")

        path = self._path_for(effective_slug)
        lock = self._lock_for(path)
        with lock:
            existing_raw = self._read_raw(path)
            if existing_raw is not None:
                existing = _validate_payload(existing_raw)
                created = existing.created
                name = name.strip() or existing.name
            else:
                created_iso = _now_iso()
                created = _parse_iso(created_iso)

            modified_iso = _now_iso()
            data = {
                "name": name.strip(),
                "slug": effective_slug,
                "created": created.isoformat().replace("+00:00", "Z"),
                "modified": modified_iso,
                "items": [item.model_dump() for item in items],
            }
            self._atomic_write(path, data)

        return CanvasSaveResponse(
            slug=effective_slug,
            modified=_parse_iso(modified_iso),
        )

    def delete(self, slug: str) -> bool:
        """
        Delete a canvas by slug. Returns True if a file was removed,
        False if no such canvas existed.
        """
        path = self._path_for(slug)
        lock = self._lock_for(path)
        with lock:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
        # Best-effort: clean up any orphaned .tmp files for this slug.
        for leftover in self.canvas_dir.glob(f"{slug}.json.tmp.*"):
            try:
                leftover.unlink()
            except OSError:
                pass
        return True

    def export_markdown(self, slug: str) -> tuple[str, str] | None:
        """
        Render a canvas to markdown. Returns (filename, content) or
        None if the slug doesn't exist.

        Each item becomes a blockquote with the vault path as a
        header, followed by the fragment body. Items are sorted by
        `order` ascending. The original fragment content comes from
        Qdrant payloads (cached at index time) — we don't read the
        vault file again on export.
        """
        detail = self.get(slug)
        if detail is None:
            return None

        sorted_items = sorted(detail.items, key=lambda i: i.order)
        lines: list[str] = [f"# {detail.name}", ""]
        for item in sorted_items:
            lines.append(f"> {item.ref}")
            lines.append("")
            if item.annotation:
                lines.append(f"_{item.annotation}_")
                lines.append("")
            lines.append("---")
            lines.append("")
        content = "\n".join(lines).rstrip() + "\n"
        return f"{detail.slug}.md", content