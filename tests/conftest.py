"""
tests/conftest.py

Shared pytest fixtures:
  - test mode env vars (env defaults so config.load() succeeds
    without a real .env)
  - in-memory Qdrant client wired into QdrantFragments
  - a mock Ollama client (records calls, returns canned chat /
    embed responses)
  - a fake vault directory (quotations/, facts/, thoughts/)
  - a TestClient for the FastAPI app with the above wired in
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure repo root is on sys.path so `import search` works without
# the package being installed. Tests should work without
# `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Set env vars BEFORE any search.* import so config.load() sees them.
# In test mode QDRANT_API_KEY is irrelevant — we wire an in-memory
# Qdrant directly via the fixtures. CANVAS_DIR is overridden per
# test via tmp_path.
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_COLLECTION", "vault-fragments-test")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("OLLAMA_CHAT_MODEL", "qwen2.5:14b")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBED_DIM", "768")
os.environ.setdefault("TOP_K", "8")
os.environ.setdefault("MAX_TOOL_CALLS", "6")


# ---------------------- Mock Ollama client ----------------------


class MockOllama:
    """
    Drop-in replacement for search.ollama_client.OllamaClient.

    Records every call to chat() and embed_one()/embed(). The test
    sets `next_chat_response` to control what /v1/chat/completions
    returns next, and `next_embed_vectors` for /v1/embeddings.

    The defaults produce a chat response with no tool_calls (final
    reply) and a single zero vector for embeddings — enough for
    smoke tests that don't care about specific values.
    """

    def __init__(self):
        self.chat_calls: list[dict] = []
        self.embed_calls: list[list[str]] = []
        # Stack of chat responses. Each pop returns one. If the
        # stack is empty, return a default final-reply response.
        self._chat_responses: list[dict] = []
        # Stack of embed results. Each pop returns one list[float].
        # If empty, return a single zero vector of EMBED_DIM.
        self._embed_results: list[list[list[float]]] = []
        self.healthz_result = True
        # Surface a specific exception on the next call. Tests
        # set this before the call and the call consumes it.
        self.fail_next_with: Exception | None = None
        self.embed_dim_default = int(os.environ.get("EMBED_DIM", "768"))

    def push_chat(self, response: dict) -> None:
        self._chat_responses.append(response)

    def push_embed(self, vectors: list[list[float]]) -> None:
        self._embed_results.append(vectors)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        self.chat_calls.append({"messages": messages, "tools": tools, "model": model})
        if self.fail_next_with is not None:
            err = self.fail_next_with
            self.fail_next_with = None
            raise err
        if self._chat_responses:
            return self._chat_responses.pop(0)
        # Default: a final assistant reply with no tool calls.
        return {
            "id": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Here are fragments matching your query.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        if self.fail_next_with is not None:
            err = self.fail_next_with
            self.fail_next_with = None
            raise err
        if self._embed_results:
            return self._embed_results.pop(0)
        # Default: one zero vector per text.
        return [[0.0] * self.embed_dim_default for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        vecs = self.embed([text])
        return vecs[0] if vecs else []

    def healthz(self) -> bool:
        return self.healthz_result

    @property
    def chat_url(self) -> str:
        return "http://mock-ollama/v1/chat/completions"

    @property
    def embed_url(self) -> str:
        return "http://mock-ollama/v1/embeddings"


@pytest.fixture
def mock_ollama() -> MockOllama:
    """Fresh MockOllama per test."""
    return MockOllama()


# ---------------------- In-memory Qdrant ----------------------


@pytest.fixture
def qdrant_in_memory():
    """
    A QdrantFragments wrapping an in-memory QdrantClient, fresh
    per test. Creates the collection on first use.
    """
    from qdrant_client import QdrantClient
    from search.qdrant_client import QdrantFragments

    client = QdrantClient(location=":memory:")
    qd = QdrantFragments(client=client, collection="vault-fragments-test")
    qd.ensure_collection(768)
    return qd


# ---------------------- Fake vault ----------------------


@pytest.fixture
def fake_vault(tmp_path: Path) -> Path:
    """
    Create a fake vault with the three canonical fragment dirs
    and a few .md files. Returns the vault root path.
    """
    vault = tmp_path / "vault"
    (vault / "quotations").mkdir(parents=True)
    (vault / "facts").mkdir(parents=True)
    (vault / "thoughts").mkdir(parents=True)
    (vault / "quotations" / "Calasso_Page42.md").write_text(
        "Sacrifice is the paradigm of all exchange.", encoding="utf-8"
    )
    (vault / "facts" / "Capitals_2024.md").write_text(
        "The Nile flooded in late summer in ancient Egypt.", encoding="utf-8"
    )
    (vault / "thoughts" / "On_ritual.md").write_text(
        "What survives of ritual when its gods are gone?", encoding="utf-8"
    )
    # An empty file to test the empty-content skip path.
    (vault / "quotations" / "empty.md").write_text("", encoding="utf-8")
    # A nested file to test recursive walking.
    nested = vault / "thoughts" / "sub"
    nested.mkdir()
    (nested / "deep.md").write_text("A deep thought.", encoding="utf-8")
    return vault


# ---------------------- Test client + config ----------------------


@pytest.fixture
def app_config(tmp_path: Path, monkeypatch, fake_vault: Path) -> "search.config.Config":
    """Build a config rooted at the fake vault and a tmp canvas dir."""
    from search import config as config_mod

    canvas_dir = tmp_path / "canvases"
    canvas_dir.mkdir()
    monkeypatch.setenv("VAULT_PATH", str(fake_vault))
    monkeypatch.setenv("CANVAS_DIR", str(canvas_dir))
    return config_mod.load()


@pytest.fixture
def canvas_store(app_config) -> "search.canvas.CanvasStore":
    from search import canvas as canvas_mod
    return canvas_mod.CanvasStore(canvas_dir=app_config.canvas_dir)


@pytest.fixture
def app_client(
    app_config,
    qdrant_in_memory,
    mock_ollama,
    canvas_store,
    monkeypatch,
):
    """
    Build a FastAPI app with the in-memory Qdrant, mock Ollama,
    and tmp canvas store wired in. Returns a TestClient.
    """
    # Reset module-level state so create_app() rebuilds cleanly.
    from search import app as app_mod
    app_mod.reset_for_tests()

    from search.app import create_app
    app = create_app(
        cfg=app_config,
        qdrant=qdrant_in_memory,
        ollama=mock_ollama,
        canvas_store=canvas_store,
    )
    # Reset again so the next test doesn't reuse these fixtures.
    yield app
    app_mod.reset_for_tests()