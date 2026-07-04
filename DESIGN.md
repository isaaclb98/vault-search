# Design — `vault-search`

> Big-picture design document. Implementation lives in
> `docs/IMPLEMENTATION_DESIGN.md` (to be written before any code is
> written).

## What this is

A web app for retrieving and presenting embedded text fragments from
an Obsidian vault. The operator talks to an LLM; the LLM finds
fragments via vault-vector tools; the operator drags fragments onto a
canvas and arranges them into a narrative order. Saved canvases
persist as JSON and export to markdown.

## Substrate

Obsidian vault with three fragment types:

- `quotations/` — sourced quotes from reading
- `facts/` — observations about the world
- `thoughts/` — own reflections

Each `.md` file is one fragment (atomic unit). Fragment identity is
the vault-relative filename (e.g.,
`quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md`).

## Primary job

Fragment-oriented retrieval and presentation. The app's purpose is to
find, see, and arrange fragments. It is not a writing tool — the
fragments are the input, and the arranged canvas is the output.

## Core loop

1. Open the chat panel.
2. Ask the LLM for fragments: by topic, by source, by contradiction.
3. The LLM calls vault-vector tools (`query`, `antipode`, `divergence`,
   `bridge`, `constellate`, `suture`, `topology`) to find candidates.
4. The LLM responds with fragments rendered as draggable cards.
5. The operator drags interesting fragments onto the canvas.
6. The operator reorders fragments on the canvas to compose a
   sequence.
7. The operator names and saves the canvas.
8. The operator exports the canvas as markdown.

## Architecture

```
                     ┌────────────────────────────────────────┐
                     │            Browser                     │
                     │  ┌──────────────┐    ┌──────────────┐  │
                     │  │  Chat Panel  │    │    Canvas    │  │
                     │  │  (LLM tools) │    │  (drag/drop) │  │
                     │  └──────┬───────┘    └──────┬───────┘  │
                     └─────────┼───────────────────┼──────────┘
                               │ HTTP              │ HTTP
                               ▼                   ▼
                     ┌────────────────────────────────────────┐
                     │  vault-search (FastAPI + Jinja SSR)     │
                     │  ┌──────────────┐    ┌──────────────┐   │
                     │  │  /api/chat   │    │ /api/canvas  │   │
                     │  │  + tools     │    │ (CRUD + ext.)│   │
                     │  └──────┬───────┘    └──────┬───────┘   │
                     └─────────┼───────────────────┼──────────┘
                               │                   │
                               │                   ▼
                               │           ┌────────────────┐
                               │           │ canvases/*.json│
                               │           └────────────────┘
                               ▼
                     ┌────────────────────────────────────────┐
                     │  vault-vector (in-process module)       │
                     │  query / antipode / divergence /        │
                     │  bridge / constellate / suture /        │
                     │  topology                              │
                     └────────────┬───────────────────────────┘
                                  │ Qdrant client
                                  ▼
                     ┌────────────────────────────────────────┐
                     │  Qdrant (operator-hosted instance)      │
                     │  collection: vault-fragments (768d)     │
                     └────────────────────────────────────────┘

                     ┌────────────────────────────────────────┐
                     │  Ollama (local)                         │
                     │  qwen2.5:14b (chat)                     │
                     │  nomic-embed-text (embeddings)          │
                     └────────────────────────────────────────┘
```

### Components

- **vault-vector (separate project):** Tool layer. Wraps Qdrant ops
  as LLM-callable functions.
- **This web app (`vault-search`):** Visual surface. Chat panel,
  canvas, persistence.
- **Qdrant (hosted, operator-owned):** Vector store. Single
  collection `vault-fragments`.
- **Ollama (local):** LLM + embeddings. Models: `qwen2.5:14b` for
  chat, `nomic-embed-text` for embeddings.

### Data flow

A message enters the chat panel, becomes an LLM prompt with the
vector ops as tools. The LLM emits tool calls; the in-process tool
implementations hit Qdrant and return fragments. The LLM synthesizes
its answer with citations. The chat panel renders fragment cards;
each card is draggable to the canvas. The canvas state lives in JSON
files under `canvases/` and is rendered into a list of
fragments-with-positions for the next page render.

## Tech stack

Mirrors the existing image-search project:

- Python 3.11+
- FastAPI + Jinja2 SSR
- Qdrant client
- Ollama via local HTTP API
- `uv` / `pip` + `pyproject.toml`
- `pytest` + `httpx` for tests

## MVP scope (v0)

- Chat panel (LLM-mediated fragment retrieval)
- Fragment cards as draggable objects
- Canvas (drag, drop, reorder, delete)
- Save canvas (named, persisted as JSON in `canvases/`)
- Export canvas to markdown

## Deferred to v1+

- Multiple-canvas library (named, dated)
- Map view (UMAP scatter)
- Traverse mode (linear walk through embedding space)
- Standalone search
- LLM as composition assistant (suggested arrangements)
- Random resurface
- Discover / Tinder-style feedback
- Favorites
- Annotation between fragments on canvas
- Obsidian-style wikilink resolution across canvas and chat

## What this is NOT

- A search engine (the LLM is the search; results are fragments)
- A writing tool (no prose generation, no draft composition)
- A knowledge graph (no entity extraction, no relationship modeling)
- A taxonomy/organizer (no tags, no categories beyond the three
  fragment types)
- A multi-user tool (single-user)
- A cloud-dependent tool (everything runs locally except Qdrant,
  which is the operator's own instance)

## Privacy

- All LLM inference local (Ollama)
- All embedding generation local (Ollama)
- Qdrant is hosted but operator-owned
- Vault files never leave the local machine
- No telemetry, no analytics

## Boundaries

- `vault-search` (this repo) is the visual surface.
- `vault-vector` (separate repo) is the tool layer.
- v0 ships with no `vault-vector` server; the chat panel calls
  vector ops as Python functions in-process. A network-call layer is
  v1+ work.
