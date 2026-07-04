# vault-search

Web app for retrieving and arranging fragments from an Obsidian
vault. A chat panel talks to a local LLM; the LLM finds fragments
via semantic search; the operator drags fragment cards onto a
canvas, reorders them, names the canvas, and exports to markdown.

> **Status:** v0 — minimal viable product. Chat + canvas + save +
> export only. See "What's deferred to v1+" below.

## Design

- **[DESIGN.md](DESIGN.md)** — big-picture: topology, philosophy,
  boundaries, what's intentionally out of scope.
- **[docs/IMPLEMENTATION_DESIGN.md](docs/IMPLEMENTATION_DESIGN.md)** —
  fine-grained: module layout, function signatures, full
  request/response contracts, error matrix, performance budgets,
  and the full UI test specification.

Read `DESIGN.md` first. The implementation doc is its companion and
answers the "how" questions.

## Repository layout

```
vault-search/
├── DESIGN.md, docs/                  # design docs
├── pyproject.toml                    # deps
├── search/                           # FastAPI app (web surface)
├── indexer/                          # CLI (one-shot vault ingest)
├── tests/                            # pytest + UI test spec
├── canvases/                         # saved canvas JSON files
├── .env.example                      # documented env vars
└── README.md                         # this file
```

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy and fill in secrets.
cp .env.example .env

# Start Ollama (separate service; not part of this repo).
ollama serve &
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# One-shot: walk your vault, embed each fragment, push to Qdrant.
# Re-running is idempotent — same point IDs are upserted, no
# duplicates.
python -m indexer.indexer

# Run the search app.
uvicorn search.app:_build_default_app --factory \
    --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

### Environment variables

All config is via environment (loaded from `.env` if present). See
[`.env.example`](.env.example) for the full list. The required
ones are `VAULT_PATH`, `QDRANT_URL`, `QDRANT_API_KEY`,
`QDRANT_COLLECTION`, `OLLAMA_HOST`, `OLLAMA_CHAT_MODEL`, and
`OLLAMA_EMBED_MODEL`. `CANVAS_DIR` defaults to `./canvases`.

## Docker deployment

Not provided in v0. The repository ships a plain FastAPI app — run
it under any WSGI/ASGI host (uvicorn, gunicorn, systemd-nspawn,
Docker, etc.) once Qdrant and Ollama are reachable. Wire your
`.env` through however you ship secrets.

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests use:

- an **in-memory Qdrant** (`QdrantClient(location=':memory:')`) — no
  network, no real Qdrant instance needed;
- a **mock Ollama client** (`tests/conftest.py::MockOllama`) that
  records calls and returns canned chat / embed responses;
- a **fake vault** built in `tmp_path` with the three canonical
  fragment directories (`quotations/`, `facts/`, `thoughts/`);
- `httpx` is in the `dev` extras so `fastapi.testclient.TestClient`
  exercises the full HTTP layer against the in-memory stack.

UI test cases are in [`tests/test_ui.spec.md`](tests/test_ui.spec.md)
— a hand-runnable checklist for v0, with `[AUTO]` tags for
Playwright coverage in a future revision.

## What's in v0

1. ✅ Chat panel: ask the LLM for fragments, get fragment cards
   back.
2. ✅ Fragment cards: draggable, with type chip + preview + source
   path.
3. ✅ Canvas: drag, drop, reorder, delete.
4. ✅ Save canvas: named, persisted as JSON in `canvases/`.
5. ✅ Export canvas to markdown (`<slug>.md`).
6. ✅ Indexer: walk `VAULT_PATH/{quotations,facts,thoughts}/`,
   embed with `nomic-embed-text`, push to `vault-fragments`
   collection. Idempotent.
7. ✅ Fragment fetch: `GET /api/fragment/<path:ref>` returns full
   content for the canvas drop handler.

Templates: `/`. JSON API at `/api/*`. Health check at `/healthz`.

## Deviations from IMPLEMENTATION_DESIGN.md

- **Qdrant point ID encoding.** The design doc specifies point IDs
  as the vault-relative path string (e.g.
  `quotations/Calasso_Page42.md`). `qdrant-client>=1.10` enforces
  UUID or unsigned-integer point IDs; raw strings are rejected with
  `ValueError: Point id ... is not a valid UUID`. To stay compliant
  with the Qdrant schema while preserving idempotency (same path
  → same point) and the public ref-as-path contract, the wrapper
  derives a deterministic UUID5 from the path via
  `uuid.uuid5(NAMESPACE_URL, ref)`. The payload's `path` field
  always carries the canonical ref, so reads round-trip back to the
  vault-relative path. This is a one-line implementation detail;
  the public API (`FragmentHit.id`, `ChatFragment.id`,
  `/api/fragment/<ref>`) still uses the path string throughout.

## What's deferred to v1+

- Multiple-canvas library page (named, dated canvases browseable
  from the UI).
- Map view (UMAP scatter of fragments).
- Traverse mode (linear walk through embedding space).
- Standalone search (no LLM in the loop).
- LLM as composition assistant (suggested arrangements on the
  canvas).
- Random resurface / Discover feed / Tinder-style feedback.
- Favorites / per-fragment likes.
- Annotation between fragments on the canvas (the data model
  already has an `annotation` field per item — v0 just doesn't
  expose editing yet).
- Obsidian-style wikilink resolution across canvas and chat.
- Streaming responses (SSE) for the chat pipeline.
- Drag-drop library (SortableJS) if vanilla HTML5 becomes painful.
- Auth if exposed beyond localhost.
- Per-canvas file lock replaced with real (cross-process) locking.
- Additional vault-vector tools beyond `query` (antipode,
  divergence, bridge, constellate, suture, topology) wired into the
  LLM tool surface.