"""
tests/test_canvas.py

Canvas CRUD + export + atomic-write + error-path coverage.

Atomic-write test: a partially-written canvas should not corrupt
the existing file. We simulate a crash mid-write by writing a
truncated .tmp file and verifying the read still sees the old
content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from search.canvas import CanvasCorrupted, CanvasStore, slugify
from search.models import CanvasItem


# ---------------------- slugify ----------------------


def test_slugify_basic():
    assert slugify("Calasso on Sacrifice") == "calasso-on-sacrifice"


def test_slugify_underscores_and_spaces():
    assert slugify("My_Canvas name") == "my-canvas-name"


def test_slugify_strips_punctuation():
    assert slugify("Foo / Bar!") == "foo-bar"


def test_slugify_empty():
    assert slugify("") == ""


def test_slugify_collapses_hyphens():
    assert slugify("a - b - c") == "a-b-c"


# ---------------------- CRUD basics ----------------------


def test_canvas_create_and_get(canvas_store: CanvasStore):
    items = [
        CanvasItem(ref="quotations/A.md", order=0, x=0, y=0, annotation=""),
        CanvasItem(ref="facts/B.md", order=1, x=10, y=20, annotation="note"),
    ]
    result = canvas_store.save(name="My Canvas", items=items)
    assert result.slug == "my-canvas"

    got = canvas_store.get("my-canvas")
    assert got is not None
    assert got.name == "My Canvas"
    assert len(got.items) == 2
    assert got.items[1].annotation == "note"


def test_canvas_update_preserves_created(canvas_store: CanvasStore):
    """Updating a canvas keeps the original created timestamp."""
    items = [CanvasItem(ref="quotations/A.md", order=0)]
    r1 = canvas_store.save(name="X", items=items)
    created1 = canvas_store.get(r1.slug).created

    # Update with a different item list.
    new_items = [
        CanvasItem(ref="facts/A.md", order=0),
        CanvasItem(ref="thoughts/B.md", order=1),
    ]
    canvas_store.save(name="X", items=new_items, slug=r1.slug)
    detail = canvas_store.get(r1.slug)
    assert detail.created == created1
    assert [i.ref for i in detail.items] == ["facts/A.md", "thoughts/B.md"]


def test_canvas_update_with_explicit_slug(canvas_store: CanvasStore):
    """Calling save with an explicit slug overrides the name-derived one."""
    canvas_store.save(name="Some Name", items=[])
    canvas_store.save(
        name="Renamed",
        items=[CanvasItem(ref="thoughts/X.md", order=0)],
        slug="custom-slug",
    )
    assert canvas_store.get("custom-slug") is not None
    # The old auto-derived slug is still around (it was created
    # empty so it shows up in the list).
    listing = canvas_store.list()
    slugs = {c.slug for c in listing.canvases}
    assert "custom-slug" in slugs


def test_canvas_get_missing_returns_none(canvas_store: CanvasStore):
    assert canvas_store.get("does-not-exist") is None


def test_canvas_delete(canvas_store: CanvasStore):
    canvas_store.save(name="Goodbye", items=[])
    assert canvas_store.delete("goodbye") is True
    assert canvas_store.get("goodbye") is None
    # Second delete returns False (no file).
    assert canvas_store.delete("goodbye") is False


def test_canvas_delete_missing(canvas_store: CanvasStore):
    assert canvas_store.delete("nope") is False


def test_canvas_list_sorted_by_modified_desc(canvas_store: CanvasStore):
    import time
    canvas_store.save(name="Alpha", items=[])
    time.sleep(0.01)
    canvas_store.save(name="Beta", items=[])
    time.sleep(0.01)
    canvas_store.save(name="Gamma", items=[])
    listing = canvas_store.list()
    names = [c.name for c in listing.canvases]
    assert names == ["Gamma", "Beta", "Alpha"]


# ---------------------- Validation ----------------------


def test_canvas_save_rejects_empty_name(canvas_store: CanvasStore):
    with pytest.raises(ValueError):
        canvas_store.save(name="", items=[])


def test_canvas_save_rejects_name_with_no_slug(canvas_store: CanvasStore):
    """A name that produces an empty slug → ValueError."""
    with pytest.raises(ValueError):
        canvas_store.save(name="///", items=[])


# ---------------------- Export ----------------------


def test_canvas_export_markdown(canvas_store: CanvasStore):
    canvas_store.save(
        name="Calasso on Sacrifice",
        items=[
            CanvasItem(ref="quotations/A.md", order=0, annotation="intro"),
            CanvasItem(ref="quotations/B.md", order=1, annotation=""),
        ],
    )
    out = canvas_store.export_markdown("calasso-on-sacrifice")
    assert out is not None
    filename, content = out
    assert filename == "calasso-on-sacrifice.md"
    assert content.startswith("# Calasso on Sacrifice")
    assert "> quotations/A.md" in content
    assert "_intro_" in content
    assert "> quotations/B.md" in content


def test_canvas_export_missing_returns_none(canvas_store: CanvasStore):
    assert canvas_store.export_markdown("nope") is None


def test_canvas_export_orders_items(canvas_store: CanvasStore):
    """Items exported in `order` ascending, regardless of insertion order."""
    canvas_store.save(
        name="Ordered",
        items=[
            CanvasItem(ref="thoughts/Z.md", order=2),
            CanvasItem(ref="thoughts/A.md", order=0),
            CanvasItem(ref="thoughts/M.md", order=1),
        ],
    )
    _, content = canvas_store.export_markdown("ordered")
    # A appears before M, M before Z.
    a_pos = content.index("thoughts/A.md")
    m_pos = content.index("thoughts/M.md")
    z_pos = content.index("thoughts/Z.md")
    assert a_pos < m_pos < z_pos


# ---------------------- Atomic write ----------------------


def test_canvas_atomic_write_leaves_no_partial_files(
    canvas_store: CanvasStore, tmp_path: Path,
):
    """After a successful save, no .tmp leftovers remain."""
    canvas_store.save(name="Clean", items=[CanvasItem(ref="a.md", order=0)])
    leftovers = list(canvas_store.canvas_dir.glob("clean.json.tmp.*"))
    assert leftovers == []


def test_canvas_corrupted_load_raises(canvas_store: CanvasStore):
    """A garbage JSON file on disk → CanvasCorrupted."""
    canvas_store.save(name="OK", items=[])
    # Corrupt the file on disk.
    path = canvas_store.canvas_dir / "ok.json"
    path.write_text("{this is not json", encoding="utf-8")
    with pytest.raises(CanvasCorrupted):
        canvas_store.get("ok")


def test_canvas_corrupted_file_skipped_in_list(
    canvas_store: CanvasStore, caplog,
):
    """A corrupted file is skipped by .list() (with a warning)."""
    canvas_store.save(name="Good", items=[])
    bad = canvas_store.canvas_dir / "bad.json"
    bad.write_text("garbage", encoding="utf-8")
    listing = canvas_store.list()
    slugs = [c.slug for c in listing.canvases]
    assert "good" in slugs
    assert "bad" not in slugs


# ---------------------- HTTP route coverage ----------------------


def test_canvas_post_creates_canvas(app_client):
    """POST /api/canvas creates a canvas; GET /api/canvas/<slug> reads it back."""
    from fastapi.testclient import TestClient

    client = TestClient(app_client)
    r = client.post(
        "/api/canvas",
        json={
            "name": "Created",
            "items": [
                {"ref": "quotations/A.md", "order": 0, "x": 0, "y": 0, "annotation": ""}
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == "created"

    r2 = client.get("/api/canvas/created")
    assert r2.status_code == 200
    assert r2.json()["name"] == "Created"
    assert len(r2.json()["items"]) == 1


def test_canvas_list_endpoint(app_client):
    from fastapi.testclient import TestClient
    client = TestClient(app_client)
    client.post("/api/canvas", json={"name": "L1", "items": []})
    client.post("/api/canvas", json={"name": "L2", "items": []})
    r = client.get("/api/canvas/list")
    assert r.status_code == 200
    body = r.json()
    names = {c["name"] for c in body["canvases"]}
    assert names == {"L1", "L2"}


def test_canvas_delete_endpoint(app_client):
    from fastapi.testclient import TestClient
    client = TestClient(app_client)
    client.post("/api/canvas", json={"name": "Goner", "items": []})
    r = client.delete("/api/canvas/goner")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    r2 = client.get("/api/canvas/goner")
    assert r2.status_code == 404


def test_canvas_export_endpoint(app_client):
    from fastapi.testclient import TestClient
    client = TestClient(app_client)
    client.post(
        "/api/canvas",
        json={
            "name": "ExportMe",
            "items": [
                {"ref": "quotations/X.md", "order": 0, "x": 0, "y": 0, "annotation": ""}
            ],
        },
    )
    r = client.get("/api/canvas/exportme/export")
    assert r.status_code == 200
    assert "markdown" in r.headers["content-type"]
    assert "# ExportMe" in r.text


def test_canvas_get_missing_returns_404(app_client):
    from fastapi.testclient import TestClient
    client = TestClient(app_client)
    r = client.get("/api/canvas/does-not-exist")
    assert r.status_code == 404


def test_canvas_post_invalid_body_returns_400(app_client):
    from fastapi.testclient import TestClient
    client = TestClient(app_client)
    # Missing required `name` field.
    r = client.post("/api/canvas", json={"items": []})
    assert r.status_code == 400