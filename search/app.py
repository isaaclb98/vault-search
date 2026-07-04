"""
search/app.py — FastAPI factory.

Routes:
  GET  /                         main page (chat panel + canvas)
  POST /api/chat                 LLM-mediated fragment retrieval
  POST /api/canvas               create or update a canvas
  GET  /api/canvas/<slug>        fetch a canvas (JSON)
  DELETE /api/canvas/<slug>      delete a canvas
  GET  /api/canvas/<slug>/export export canvas to markdown
  GET  /api/canvas/list          list all canvases (JSON)
  GET  /api/fragment/<path:ref>  fetch full fragment content by vault-relative path
  GET  /healthz                  liveness check

Design reference: IMPLEMENTATION_DESIGN.md §"Canvas pipeline /
Endpoints" and §"Error matrix".
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from search import canvas as canvas_mod
from search import chat as chat_mod
from search import config as config_mod
from search import fragment_resolver as fragment_resolver_mod
from search import ollama_client as ollama_client_mod
from search import qdrant_client as qdrant_client_mod
from search.models import (
    CanvasDeleteResponse,
    CanvasListResponse,
    CanvasSaveRequest,
    CanvasSaveResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    FragmentResponse,
)

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

# Bump this when you change any static asset (CSS, JS). It's appended
# as ?v=N in the templates to force browsers to re-fetch.
STATIC_ASSETS_VERSION: int = 1


# ---------------------- App state ----------------------

# Module-level singletons, set by create_app(). Tests can swap them
# via dependency-injection arguments to create_app() before calling
# the test client.
_cfg: config_mod.Config | None = None
_qdrant: qdrant_client_mod.QdrantFragments | None = None
_ollama: ollama_client_mod.OllamaClient | None = None
_canvas_store: canvas_mod.CanvasStore | None = None
_templates: Jinja2Templates | None = None


def get_cfg() -> config_mod.Config:
    if _cfg is None:
        raise RuntimeError("Config not initialized — call create_app() first")
    return _cfg


def get_qdrant() -> qdrant_client_mod.QdrantFragments:
    if _qdrant is None:
        raise RuntimeError("QdrantFragments not initialized — call create_app() first")
    return _qdrant


def get_ollama() -> ollama_client_mod.OllamaClient:
    if _ollama is None:
        raise RuntimeError("OllamaClient not initialized — call create_app() first")
    return _ollama


def get_canvas_store() -> canvas_mod.CanvasStore:
    if _canvas_store is None:
        raise RuntimeError("CanvasStore not initialized — call create_app() first")
    return _canvas_store


def reset_for_tests() -> None:
    """Drop module state so the next create_app() rebuilds. Test-only."""
    global _cfg, _qdrant, _ollama, _canvas_store, _templates
    _cfg = None
    _qdrant = None
    _ollama = None
    _canvas_store = None
    _templates = None


# ---------------------- Error helpers ----------------------


def _err(status: int, error: str, detail: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=error, detail=detail, code=code).model_dump(),
    )


def _llm_down(detail: str) -> JSONResponse:
    return _err(503, "ollama_unavailable", detail, "ollama_unavailable")


def _llm_missing(detail: str) -> JSONResponse:
    return _err(503, "ollama_model_missing", detail, "ollama_model_missing")


def _qdrant_down(detail: str) -> JSONResponse:
    return _err(503, "qdrant_unavailable", detail, "qdrant_unavailable")


def _internal(detail: str) -> JSONResponse:
    return _err(500, "internal_error", detail, "internal_error")


# ---------------------- App factory ----------------------


def create_app(
    cfg: config_mod.Config | None = None,
    qdrant: qdrant_client_mod.QdrantFragments | None = None,
    ollama: ollama_client_mod.OllamaClient | None = None,
    canvas_store: canvas_mod.CanvasStore | None = None,
    templates: Jinja2Templates | None = None,
) -> FastAPI:
    """
    Build a FastAPI app with all routes wired.

    Args:
        cfg: pre-loaded config (defaults to config.load())
        qdrant: pre-built QdrantFragments (default: built from cfg)
        ollama: pre-built OllamaClient (default: built from cfg)
        canvas_store: pre-built CanvasStore (default: built from cfg)
        templates: pre-built Jinja2Templates (default: built from
            search/templates)
    """
    global _cfg, _qdrant, _ollama, _canvas_store, _templates
    _cfg = cfg or config_mod.load()
    logging.basicConfig(level=_cfg.log_level)

    if qdrant is None:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=_cfg.qdrant_url,
            api_key=_cfg.qdrant_api_key,
            timeout=30,
        )
        qdrant = qdrant_client_mod.QdrantFragments(
            client=client, collection=_cfg.qdrant_collection,
        )
    _qdrant = qdrant

    if ollama is None:
        ollama = ollama_client_mod.OllamaClient(
            host=_cfg.ollama_host,
            chat_model=_cfg.ollama_chat_model,
            embed_model=_cfg.ollama_embed_model,
        )
    _ollama = ollama

    if canvas_store is None:
        canvas_store = canvas_mod.CanvasStore(canvas_dir=_cfg.canvas_dir)
    _canvas_store = canvas_store

    templates = templates or Jinja2Templates(directory=str(TEMPLATES_DIR))
    _templates = templates

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Best-effort warm-up checks. Don't crash the app if either
        # backend is down at startup — let the routes surface the
        # error per request.
        if not qdrant.healthz():
            logger.warning(
                "Qdrant unreachable at startup (%s) — chat will fail until it recovers",
                _cfg.qdrant_url,
            )
        yield

    app = FastAPI(
        title="vault-search",
        version="0.1.0",
        lifespan=lifespan,
        # Don't redirect /canvas to /canvas/ etc. — the URL paths
        # carry meaning (a trailing slash on the slug endpoint would
        # look like a different slug).
        redirect_slashes=False,
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---------------------- Routes ----------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "static_assets_version": STATIC_ASSETS_VERSION,
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "qdrant": qdrant.healthz(),
            "ollama": ollama.healthz(),
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: Request) -> JSONResponse:
        """
        LLM-mediated fragment retrieval. Body is a ChatRequest.

        Error matrix:
          400  — invalid request body
          503  — Ollama unreachable or model missing
          503  — Qdrant unreachable
        """
        try:
            body = await request.json()
            req = ChatRequest.model_validate(body)
        except Exception as e:
            return _err(400, "bad_request", str(e), "bad_request")

        history_dicts = [m.model_dump() for m in req.history]
        try:
            result = chat_mod.run_chat_loop(
                cfg=_cfg,
                ollama=ollama,
                qdrant=qdrant,
                history=history_dicts,
                message=req.message,
            )
        except ollama_client_mod.OllamaUnavailable as e:
            return _llm_down(str(e))
        except ollama_client_mod.OllamaModelMissing as e:
            return _llm_missing(
                f"Model not loaded. Run `ollama pull {_cfg.ollama_chat_model}` "
                f"and `ollama pull {_cfg.ollama_embed_model}`. Detail: {e}"
            )
        except qdrant_client_mod.QdrantUnavailable as e:
            return _qdrant_down(str(e))
        except Exception as e:
            logger.exception("chat loop crashed")
            return _internal(str(e))

        response = ChatResponse(
            reply=result.reply,
            fragments=result.fragments,
            tool_calls=result.tool_calls,
        )
        return JSONResponse(content=response.model_dump())

    @app.post("/api/canvas", response_model=CanvasSaveResponse)
    async def save_canvas(request: Request) -> JSONResponse:
        """Create or update a canvas. Body is a CanvasSaveRequest."""
        try:
            body = await request.json()
            req = CanvasSaveRequest.model_validate(body)
        except Exception as e:
            return _err(400, "bad_request", str(e), "bad_request")
        try:
            result = canvas_store.save(
                name=req.name, items=req.items, slug=req.slug,
            )
        except ValueError as e:
            return _err(400, "bad_request", str(e), "bad_request")
        except Exception as e:
            logger.exception("canvas save failed")
            return _internal(str(e))
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.get("/api/canvas/list", response_model=CanvasListResponse)
    async def list_canvases() -> JSONResponse:
        return JSONResponse(content=canvas_store.list().model_dump(mode="json"))

    @app.get("/api/canvas/{slug}")
    async def get_canvas(slug: str) -> JSONResponse:
        try:
            detail = canvas_store.get(slug)
        except canvas_mod.CanvasCorrupted as e:
            logger.exception("canvas %s corrupted", slug)
            return _err(500, "canvas_corrupted", str(e), "canvas_corrupted")
        if detail is None:
            return _err(404, "not_found", f"no canvas with slug {slug!r}", "not_found")
        return JSONResponse(content=detail.model_dump(mode="json"))

    @app.delete("/api/canvas/{slug}", response_model=CanvasDeleteResponse)
    async def delete_canvas(slug: str) -> JSONResponse:
        deleted = canvas_store.delete(slug)
        if not deleted:
            return _err(404, "not_found", f"no canvas with slug {slug!r}", "not_found")
        return JSONResponse(
            content=CanvasDeleteResponse(slug=slug).model_dump()
        )

    @app.get("/api/canvas/{slug}/export")
    async def export_canvas(slug: str) -> PlainTextResponse:
        """
        Export a canvas to markdown.

        IMPORTANT: this is a placeholder export. v0 ships the
        one-item-per-blockquote skeleton; the fragment body text
        for each item is filled in by a future revision that reads
        Qdrant payloads (or vault files) for the resolved refs. The
        skeleton output is sufficient for downstream "draft a
        document from a canvas" workflows where the consumer
        resolves refs themselves — see §"Deferred to v1+" in
        IMPLEMENTATION_DESIGN.md for the full export pipeline.
        """
        try:
            result = canvas_store.export_markdown(slug)
        except canvas_mod.CanvasCorrupted as e:
            return _err(500, "canvas_corrupted", str(e), "canvas_corrupted")
        if result is None:
            return _err(404, "not_found", f"no canvas with slug {slug!r}", "not_found")
        filename, content = result
        return PlainTextResponse(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/fragment/{ref:path}", response_model=FragmentResponse)
    async def get_fragment(ref: str) -> JSONResponse:
        """
        Fetch a fragment's full content by vault-relative path.

        Used by the canvas drop handler: the chat panel renders a
        card with the ref; on drop to the canvas, the JS hits this
        endpoint to get the full body for the canvas item.

        Returns 400 if the path is suspicious (absolute, contains
        .., not a .md file). Returns 404 if the file doesn't exist
        on disk.
        """
        try:
            resolved = fragment_resolver_mod.resolve(_cfg.vault_path, ref)
        except ValueError as e:
            return _err(400, "bad_request", str(e), "bad_request")
        if resolved is None:
            return _err(404, "not_found", f"fragment {ref!r} not found", "not_found")
        return JSONResponse(
            content=FragmentResponse(
                ref=resolved.ref,
                content=resolved.content,
                type=resolved.type,
                truncated=resolved.truncated,
            ).model_dump()
        )

    return app


def _build_default_app() -> FastAPI:
    return create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("search.app:_build_default_app", factory=True, host="0.0.0.0", port=8000)