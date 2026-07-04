"""
search/ollama_client.py

Thin wrapper around Ollama's OpenAI-compatible HTTP endpoints.

Ollama exposes /v1/chat/completions and /v1/embeddings with the same
request/response shape as the OpenAI REST API. We hit them directly
via httpx — no `openai` Python package required, no OpenAI account,
no API keys.

The client is intentionally small:
  - `embed(texts)` → list of float vectors (one per input).
  - `chat(messages, tools, model)` → OpenAI-compatible ChatResponse dict.

Errors from Ollama surface as `OllamaUnavailable` (connection refused,
DNS failure, timeout) or `OllamaError` (model not loaded, invalid
request). The chat pipeline maps these to the user-facing error
states in IMPLEMENTATION_DESIGN.md §"Error matrix".
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base class for Ollama errors. Maps to a generic 503 in the chat route."""


class OllamaUnavailable(OllamaError):
    """Ollama host is unreachable. Maps to 503 'LLM service down'."""


class OllamaModelMissing(OllamaError):
    """Ollama replied with a model-not-found error. Maps to 503 'Model X not found'."""


class OllamaClient:
    """Stateless HTTP client for Ollama's OpenAI-compatible endpoints."""

    def __init__(
        self,
        host: str,
        chat_model: str,
        embed_model: str,
        timeout: float = 30.0,
    ):
        self.host = host.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self._timeout = timeout

    @property
    def chat_url(self) -> str:
        return f"{self.host}/v1/chat/completions"

    @property
    def embed_url(self) -> str:
        return f"{self.host}/v1/embeddings"

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST JSON to Ollama. Translates connection failures to
        OllamaUnavailable, model-missing markers to OllamaModelMissing,
        everything else to OllamaError.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            raise OllamaUnavailable(
                f"Ollama unreachable at {self.host}: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise OllamaUnavailable(
                f"Ollama HTTP error from {self.host}: {e}"
            ) from e
        if resp.status_code >= 500:
            raise OllamaUnavailable(
                f"Ollama {resp.status_code} from {url}: {resp.text[:200]}"
            )
        if resp.status_code == 404:
            # Ollama returns 404 for unknown models when called via
            # the OpenAI-compatible endpoint with a model that hasn't
            # been pulled.
            raise OllamaModelMissing(
                f"Ollama returned 404 for {url}: model not loaded?"
            )
        if resp.status_code >= 400:
            # Try to extract a useful error message from the body.
            try:
                err_body = resp.json()
                err_msg = err_body.get("error", {}).get("message") or err_body.get("message") or resp.text
            except Exception:
                err_msg = resp.text
            # Heuristic: if the error mentions a model name, surface
            # it as OllamaModelMissing so the route can produce the
            # "run `ollama pull X`" message. Otherwise treat as a
            # generic Ollama error.
            if (
                self.chat_model in err_msg
                or self.embed_model in err_msg
                or "model" in err_msg.lower()
            ):
                raise OllamaModelMissing(f"Ollama error: {err_msg}")
            raise OllamaError(f"Ollama error: {err_msg}")
        try:
            return resp.json()
        except Exception as e:
            raise OllamaError(
                f"Ollama returned non-JSON from {url}: {e}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings. Returns one vector per input.

        Calls POST /v1/embeddings with `{"input": texts, "model": embed_model}`.
        OpenAI's response shape is `{"data": [{"embedding": [...]}, ...]}` —
        Ollama mirrors it. We re-order by `index` so the output is
        always input-order, even if Ollama returns out-of-order.
        """
        if not texts:
            return []
        payload = {"input": texts, "model": self.embed_model}
        data = self._post_json(self.embed_url, payload)
        rows = data.get("data") or []
        # Sort by `index` to be safe. If `index` is missing, treat as 0.
        indexed = sorted(
            rows,
            key=lambda r: (r.get("index") if isinstance(r.get("index"), int) else 0),
        )
        return [list(r.get("embedding") or []) for r in indexed]

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        vecs = self.embed([text])
        if not vecs:
            return []
        return vecs[0]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request. Returns the raw response dict.

        `messages` follows the OpenAI shape:
          [{"role": "system|user|assistant|tool", "content": "...", ...}]
        `tools` (optional) follows the OpenAI function-calling shape.
        `model` defaults to self.chat_model.

        Caller is responsible for handling the tool-call loop. The
        `chat.py` module owns that loop.

        Returned dict shape (OpenAI-compatible):
          {
            "id": "...",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "...", "tool_calls": [...]}, "finish_reason": "stop"}],
            "usage": {...}
          }
        """
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        return self._post_json(self.chat_url, payload)

    def healthz(self) -> bool:
        """Best-effort reachability check. Used by /api/canvas/list etc."""
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.host}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False