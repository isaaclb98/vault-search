# UI Test Specification — `vault-search`

**Source of truth:** `docs/IMPLEMENTATION_DESIGN.md`. If this file
and the design doc drift apart, the design doc wins.

**How to use in v0:** this is the primary UI test artifact. Work
through each case manually in a browser, marking pass/fail.

**How to use in v1+:** Cases tagged `[AUTO]` map to Playwright tests
in `tests/test_ui.py` (not present in v0). Cases tagged `[MANUAL]`
remain checklist-only.

---

## Setup

```bash
cd ~/projects/vault-search
source .venv/bin/activate
# Ollama running locally, qwen2.5:14b + nomic-embed-text pulled.
# Qdrant reachable at $QDRANT_URL with the vault-fragments
# collection populated (run `python -m indexer.indexer` once).
uvicorn search.app:_build_default_app --factory --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## UI cases

### Group A — initial page

#### UI-A-001 — initial page load, no query `[AUTO]`

- **Given:** the page is opened for the first time.
- **When:** the browser loads `GET /`.
- **Then:**
  - HTTP 200.
  - Page shows a chat panel on the left and a canvas on the right.
  - Chat panel has an empty-state message ("No messages yet…").
  - Canvas has an empty-state message ("Drag fragment cards…").
  - No network requests to `/api/chat` are made.

#### UI-A-002 — chat submit with empty textarea `[MANUAL]`

- **Given:** the user is on `GET /` with the chat panel visible.
- **When:** the user clicks "Send" without typing anything.
- **Then:**
  - The browser's built-in "Please fill out this field" tooltip
    appears on the textarea (because `required` is set).
  - No `/api/chat` request is made.

### Group B — chat flow

#### UI-B-001 — submit a message, response with fragments `[AUTO]`

- **Given:** the indexer has been run against a small fake vault.
- **When:** the user types a query and clicks "Send".
- **Then:**
  - A user message bubble appears in the chat log immediately.
  - An assistant bubble appears after the `/api/chat` response
    arrives.
  - The assistant bubble shows the reply text and one or more
    fragment cards.
  - Each fragment card has the type chip, preview text, and
    source path.
  - The textarea is cleared after the response.

#### UI-B-002 — drag a fragment card to the canvas `[AUTO]`

- **Given:** an assistant message has surfaced at least one fragment card.
- **When:** the user drags the fragment card and drops it on the
  canvas surface.
- **Then:**
  - A new canvas item appears with the fragment's type, source
    path, and body text.
  - The chat log remains unchanged.
  - The canvas status line shows "Added <ref>".

#### UI-B-003 — drag a duplicate fragment is rejected `[MANUAL]`

- **Given:** the canvas already contains fragment `quotations/A.md`.
- **When:** the user drags another `quotations/A.md` card onto the canvas.
- **Then:**
  - No new canvas item appears.
  - The canvas status line shows "Fragment already on canvas."

#### UI-B-004 — chat error surfaces inline `[MANUAL]`

- **Given:** Ollama is stopped.
- **When:** the user submits a message.
- **Then:**
  - The chat log shows an error message ("LLM service down…").
  - The textarea is re-enabled; the user can retry.
  - The page does not crash.

### Group C — canvas

#### UI-C-001 — reorder canvas items `[MANUAL]`

- **Given:** the canvas contains at least two items.
- **When:** the user drags one item onto another.
- **Then:**
  - The dragged item is moved to the dropped position.
  - The `data-order` attribute is updated on each item.
  - Subsequent save reflects the new order.

#### UI-C-002 — delete a canvas item `[AUTO]`

- **Given:** the canvas contains one or more items.
- **When:** the user clicks the × button on a canvas item.
- **Then:**
  - That item is removed from the DOM.
  - The remaining items retain their relative order.
  - The canvas empty state reappears if no items remain.

#### UI-C-003 — save a canvas `[AUTO]`

- **Given:** the canvas has a name and at least one item.
- **When:** the user clicks "Save".
- **Then:**
  - The status line shows "Saved (<slug>)".
  - The current slug is recorded for export.
  - Reloading the page preserves the canvas (via
    `/api/canvas/<slug>`).

#### UI-C-004 — save without a name is rejected `[MANUAL]`

- **Given:** the canvas has items but no name.
- **When:** the user clicks "Save".
- **Then:**
  - The status line shows "Canvas name is required."
  - No POST to `/api/canvas` is made.

#### UI-C-005 — export a saved canvas `[AUTO]`

- **Given:** a canvas has been saved and the current slug is set.
- **When:** the user clicks "Export".
- **Then:**
  - The browser downloads a `<slug>.md` file.
  - The file starts with `# <canvas name>`.
  - Each item appears as a `> <ref>` blockquote in order.

#### UI-C-006 — export before save is rejected `[MANUAL]`

- **Given:** the canvas has items but no slug yet.
- **When:** the user clicks "Export".
- **Then:**
  - The status line shows "Save the canvas first."
  - No download starts.

#### UI-C-007 — clear the canvas `[MANUAL]`

- **Given:** the canvas has items and/or a name.
- **When:** the user clicks "Clear".
- **Then:**
  - All items are removed.
  - The name input is cleared.
  - The current slug is reset.

### Group D — error matrix coverage

#### UI-D-001 — Qdrant down at chat time `[MANUAL]`

- **Given:** Qdrant is unreachable.
- **When:** the user submits a chat message that would trigger a
  `query_fragments` tool call.
- **Then:** the chat log shows a "Vector store down" error message.

#### UI-D-002 — Ollama model missing `[MANUAL]`

- **Given:** Ollama is running but `qwen2.5:14b` has not been pulled.
- **When:** the user submits a chat message.
- **Then:** the chat log shows a "Model X not found, run `ollama pull X`"
  error message.

### Group E — fragment fetch

#### UI-E-001 — fetch full fragment by ref `[AUTO]`

- **Given:** a fragment file exists at
  `quotations/Calasso_Page42.md` in the vault.
- **When:** the browser hits
  `GET /api/fragment/quotations/Calasso_Page42.md`.
- **Then:**
  - HTTP 200.
  - The JSON body contains `ref`, `content`, `type`, and
    `truncated=false`.

#### UI-E-002 — fetch missing fragment `[AUTO]`

- **Given:** no such file exists in the vault.
- **When:** the browser hits `GET /api/fragment/quotations/nope.md`.
- **Then:** HTTP 404 with a JSON error body.

#### UI-E-003 — fetch with traversal attempt `[MANUAL]`

- **Given:** an attacker (or a typo) sends a ref like
  `../../etc/passwd`.
- **When:** the browser hits `GET /api/fragment/../../etc/passwd`.
- **Then:** HTTP 400 with `bad_request` error code. No file is read.

#### UI-E-004 — fetch non-markdown file `[MANUAL]`

- **Given:** the user requests `GET /api/fragment/photos/cat.jpg`.
- **Then:** HTTP 404.