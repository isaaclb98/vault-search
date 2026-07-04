# Implementation Design — `vault-search`

> Implementation-ready design document. Read `DESIGN.md` first.

## Module layout

```
vault-search/
├── DESIGN.md
├── docs/
│   └── IMPLEMENTATION_DESIGN.md    # this file
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE                         # MIT
├── indexer/                        # one-shot ingest CLI
│   ├── __init__.py
│   └── indexer.py
├── search/                         # FastAPI app (web surface)
│   ├── __init__.py
│   ├── app.py                      # FastAPI app factory
│   ├── config.py                   # env-var loading
│   ├── qdrant_client.py            # Qdrant client + ops
│   ├── ollama_client.py            # Ollama OpenAI-compatible client
│   ├── chat.py                     # LLM chat tool loop
│   ├── canvas.py                   # canvas CRUD + export
│   ├── fragment_resolver.py        # vault filename → content
│   ├── models.py                   # Pydantic models for API
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html              # main page (chat + canvas)
│   │   └── _partials/
│   │       ├── chat_message.html
│   │       ├── fragment_card.html
│   │       └── canvas_item.html
│   └── static/
│       ├── css/site.css
│       └── js/
│           ├── chat.js
│           └── canvas.js
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_indexer.py
│   ├── test_chat.py
│   ├── test_canvas.py
│   └── test_ui.spec.md             # hand-runnable UI test cases
└── canvases/                       # saved canvas JSON files
    └── .gitkeep
```

## Configuration

All config via environment variables. Loaded with `python-dotenv`.

### Required

| Variable | Description |
|---|---|
| `VAULT_PATH` | Absolute path to Obsidian vault. Default: `~/projects/obsidian`. |
| `QDRANT_URL` | Qdrant instance URL. |
| `QDRANT_API_KEY` | Qdrant API key. |
| `QDRANT_COLLECTION` | Collection name. Default: `vault-fragments`. |
| `OLLAMA_HOST` | Ollama host URL. Default: `http://localhost:11434`. |
| `OLLAMA_CHAT_MODEL` | Chat model. Default: `qwen2.5:14b`. |
| `OLLAMA_EMBED_MODEL` | Embedding model. Default: `nomic-embed-text`. |
| `CANVAS_DIR` | Directory for canvas JSON. Default: `./canvases` (relative to repo). |

### Optional

| Variable | Default | Description |
|---|---|---|
| `EMBED_DIM` | 768 | Embedding dimension (must match model). |
| `TOP_K` | 8 | Default fragments per query. |
| `MAX_TOOL_CALLS` | 6 | Max LLM tool-call iterations per turn. |
| `LOG_LEVEL` | INFO | Standard log levels. |

## Qdrant schema

Single collection: `vault-fragments`.

**Point ID:** vault-relative path as a stable string ID (e.g., `quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md`).

**Vector:** 768-dimensional float (nomic-embed-text).

**Payload:**
```json
{
  "path": "quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md",
  "content": "the fragment body text",
  "type": "quotation|fact|thought",
  "filename": "Calasso_Ruin_of_Kasch_Page_42_Quote_12.md",
  "indexed_at": "2026-07-04T16:30:00Z"
}
```

Payload fields are filterable. No payload indexes needed for v0 (small corpus).

## Ingest pipeline

**CLI entry:** `python -m indexer.indexer`

**Behavior:**
1. Walks `VAULT_PATH/{quotations,facts,thoughts}/` recursively.
2. For each `.md` file:
   - Skip files with empty content.
   - Embed content via the configured embedding model.
   - Upsert to Qdrant with payload.
3. Prints progress (`X / N files`) to stdout.
4. Idempotent: re-running upserts over the same point IDs (no duplicates).

**Exit codes:**
- `0` — success
- `1` — Vault path not found
- `2` — Qdrant unreachable
- `3` — Embedding model unavailable

## Chat pipeline

The LLM has access to a single tool: `query_fragments`.

### Tool: `query_fragments`

```json
{
  "name": "query_fragments",
  "description": "Find fragments in the vault semantically related to the query.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Natural language search query." },
      "k": { "type": "integer", "description": "Number of fragments to return.", "default": 8 },
      "type_filter": { "type": "string", "enum": ["quotation", "fact", "thought"] }
    },
    "required": ["query"]
  }
}
```

### Endpoint: `POST /api/chat`

**Request:**
```json
{
  "message": "find me Calasso on ritual sacrifice",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are fragments on ritual sacrifice...",
  "fragments": [
    {
      "id": "quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md",
      "type": "quotation",
      "preview": "first 200 chars of content",
      "score": 0.83
    }
  ],
  "tool_calls": 2
}
```

### Server-side flow

1. Load `history` + new `message` into OpenAI-compatible chat messages.
2. Send to Ollama `/v1/chat/completions` with `tools=[query_fragments]`.
3. If response has `tool_calls`, dispatch them (call Qdrant), append results, re-send.
4. Repeat until LLM emits a final assistant message (no tool calls) or `MAX_TOOL_CALLS` reached.
5. Return final reply + the fragments used as sources.

## Canvas pipeline

### Storage

JSON files in `canvases/`, one per canvas.

**Filename:** URL-safe slug derived from canvas name (e.g., `calasso-on-sacrifice.json`).

**Schema:**
```json
{
  "name": "Calasso on Sacrifice",
  "slug": "calasso-on-sacrifice",
  "created": "2026-07-04T16:30:00Z",
  "modified": "2026-07-04T16:35:00Z",
  "items": [
    {
      "ref": "quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md",
      "order": 1,
      "x": 100,
      "y": 200,
      "annotation": ""
    },
    {
      "ref": "quotations/Calasso_Ruin_of_Kasch_Page_87_Quote_03.md",
      "order": 2,
      "x": 100,
      "y": 320,
      "annotation": "transition: from theory to example"
    }
  ]
}
```

**Concurrency:** single-user. Read-modify-write must be atomic; use a per-canvas in-process lock (sufficient for single-user). v1+ may need real locking.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Main page (chat panel + canvas) |
| `POST` | `/api/chat` | LLM-mediated fragment retrieval |
| `POST` | `/api/canvas` | Create or update a canvas |
| `GET` | `/api/canvas/<slug>` | Fetch a canvas (JSON) |
| `DELETE` | `/api/canvas/<slug>` | Delete a canvas |
| `GET` | `/api/canvas/<slug>/export` | Export to markdown |
| `GET` | `/api/canvas/list` | List all canvases (JSON) |
| `GET` | `/api/fragment/<path:ref>` | Fetch full fragment content by vault-relative path |

### Endpoint: `POST /api/canvas`

**Request:**
```json
{
  "name": "Calasso on Sacrifice",
  "items": [
    {"ref": "quotations/...", "order": 1, "x": 100, "y": 200, "annotation": ""}
  ]
}
```

**Response:** `{"slug": "calasso-on-sacrifice", "modified": "..."}`

### Endpoint: `GET /api/canvas/<slug>/export`

**Response:** `text/markdown` with fragments in order:

```markdown
# Calasso on Sacrifice

> quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md

Sacrifice is the paradigm of all exchange...

---

> quotations/Calasso_Ruin_of_Kasch_Page_87_Quote_03.md

The victim must be without blemish...

---
```

## Fragment card

Rendered in chat panel as a draggable card. Markup:

```html
<div class="fragment-card"
     draggable="true"
     data-fragment-ref="quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md">
  <div class="fragment-card__type">quotation</div>
  <div class="fragment-card__preview">first 200 chars...</div>
  <div class="fragment-card__source">quotations/Calasso_Ruin_of_Kasch_Page_42_Quote_12.md</div>
</div>
```

Drag handler in `canvas.js`: on `dragstart`, set
`dataTransfer.setData('text/fragment-ref', ref)`.

## Frontend behavior

### Chat panel

- Form with `<textarea name="message">` + submit button.
- POST to `/api/chat` (full-page response for v0 — the response HTML replaces the chat panel contents).
- Server returns HTML for new chat messages + fragment cards.

### Canvas

- 2D surface (CSS-positioned divs).
- Drop zone: `dragover` + `drop` handlers in `canvas.js`.
- Drop reads `dataTransfer.getData('text/fragment-ref')`, fetches the fragment content via `/api/fragment/<path:ref>`, renders as canvas item.
- Reorder: HTML5 drag within canvas items updates the `order` field.
- Delete: per-item delete button removes from canvas state.

### Save

- Form on the canvas page: name input + save button.
- POST to `/api/canvas` with full canvas state.

## Error matrix

| Failure | Surface | Behavior |
|---|---|---|
| Ollama unreachable | Chat | 503 page; user sees "LLM service down, try again" |
| Ollama model not loaded | Chat | 503 page; user sees "Model X not found, run `ollama pull X`" |
| Qdrant unreachable | Chat + Indexer | 503 page; user sees "Vector store down" |
| Qdrant collection missing | Chat | Auto-create on first call (or instruct user to run indexer) |
| Empty vault | Indexer | Exit 0 with warning "no .md files found" |
| Bad canvas JSON on load | Canvas | 500 page; log details, surface "Canvas corrupted" |
| Tool-call loop exceeds MAX | Chat | Return partial response, log warning |
| LLM returns invalid JSON tool call | Chat | Treat as final response, skip tool execution |

## Performance budgets

| Operation | Target | Why |
|---|---|---|
| Page load | < 200ms | Local-only render |
| Chat response (no tools) | < 3s | Local LLM, simple prompt |
| Chat response (1 tool call) | < 5s | Embedding + Qdrant query + LLM |
| Chat response (2-3 tool calls) | < 10s | Multiple round-trips |
| Canvas save | < 100ms | Single JSON write |
| Canvas export | < 200ms | File read + format |
| Indexer (per file) | < 500ms | Embedding + upsert |

## Testing

### Backend (pytest)

- `tests/test_indexer.py` — walks a fake vault, asserts Qdrant upserts (with Qdrant client mocked).
- `tests/test_chat.py` — mocks Ollama + Qdrant, asserts tool loop behavior and final response shape.
- `tests/test_canvas.py` — canvas CRUD + export, atomic writes.

### UI (hand-runnable checklist)

`tests/test_ui.spec.md` — markdown checklist of UI test cases. Hand-runnable on real services. Each case tagged `[AUTO]` (future Playwright) or `[MANUAL]` (always hand-run).

Cases:
- Open `/` → chat panel + canvas visible
- Submit message → response with fragment cards appears
- Drag fragment card to canvas → canvas item appears
- Reorder canvas item → order updates
- Save canvas → toast "saved", reload shows saved state
- Export canvas → markdown file downloads with fragments in order
- Delete canvas → canvas removed

## Open items for v1+

- Streaming responses (SSE)
- Multiple-canvas library page
- LLM as composition assistant (suggested arrangements)
- Wikilink resolution across canvas and chat
- Drag-drop library (SortableJS) if vanilla becomes painful
- Auth if exposed beyond localhost
- Frontend framework (HTMX, Alpine) if interactivity grows
- Per-canvas file lock replaced with real locking
- Additional vault-vector tools beyond `query` (antipode, divergence, bridge, constellate, suture, topology) wired into LLM tool surface