"""
search/config.py — environment variable loading + validation.

Loaded once at process start, validated up-front, frozen for the
lifetime of the process. See IMPLEMENTATION_DESIGN.md §"Configuration"
for the full table.

Why no pydantic-settings: this project uses python-dotenv + dataclass
directly so config loading stays a single file with no transitive
deps. Every var has a default except the secrets (QDRANT_API_KEY may
be empty for local dev where Qdrant has no auth).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from cwd (or any ancestor) on import. Real process env wins.
load_dotenv()

DEFAULT_VAULT_PATH: str = "~/projects/obsidian"
DEFAULT_QDRANT_COLLECTION: str = "vault-fragments"
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
DEFAULT_OLLAMA_CHAT_MODEL: str = "qwen2.5:14b"
DEFAULT_OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
DEFAULT_CANVAS_DIR: str = "./canvases"

DEFAULT_EMBED_DIM: int = 768
DEFAULT_TOP_K: int = 8
DEFAULT_MAX_TOOL_CALLS: int = 6
DEFAULT_LOG_LEVEL: str = "INFO"

# Valid fragment types for the type_filter on query_fragments and for
# the indexed-at payload discriminator. Kept as a frozenset so the
# module is import-safe (no module-level mutation).
FRAGMENT_TYPES: frozenset[str] = frozenset({"quotation", "fact", "thought"})


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"env {name}={raw!r} is not a valid int") from e


def _expand_path(raw: str) -> str:
    """Expand ~ and resolve to absolute path string for storage."""
    return str(Path(raw).expanduser().resolve())


@dataclass(frozen=True)
class Config:
    vault_path: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    ollama_host: str
    ollama_chat_model: str
    ollama_embed_model: str
    canvas_dir: str
    embed_dim: int
    top_k: int
    max_tool_calls: int
    log_level: str

    def ollama_chat_url(self) -> str:
        """OpenAI-compatible chat completions endpoint on the Ollama host."""
        return f"{self.ollama_host.rstrip('/')}/v1/chat/completions"

    def ollama_embed_url(self) -> str:
        """OpenAI-compatible embeddings endpoint on the Ollama host."""
        return f"{self.ollama_host.rstrip('/')}/v1/embeddings"


def load() -> Config:
    """
    Load config from environment. Validates required fields and
    invariants. Raises ValueError on invalid input.

    Secrets (QDRANT_API_KEY) may be empty in local dev. QDRANT_URL
    is required to be set; if unset, defaults to localhost for ease
    of dev but the chat pipeline will surface a 503 on first call.
    """
    vault_path = _expand_path(
        os.environ.get("VAULT_PATH", DEFAULT_VAULT_PATH)
    )
    if not Path(vault_path).is_dir():
        # Not fatal at config-load time — the indexer is the one that
        # needs a real vault dir, and the search app can still serve
        # empty results. The indexer validates this again.
        pass

    embed_dim = _int("EMBED_DIM", DEFAULT_EMBED_DIM)
    if embed_dim <= 0:
        raise ValueError(f"EMBED_DIM={embed_dim} must be > 0")

    top_k = _int("TOP_K", DEFAULT_TOP_K)
    if top_k <= 0:
        raise ValueError(f"TOP_K={top_k} must be > 0")

    max_tool_calls = _int("MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS)
    if max_tool_calls <= 0:
        raise ValueError(f"MAX_TOOL_CALLS={max_tool_calls} must be > 0")

    canvas_dir_raw = os.environ.get("CANVAS_DIR", DEFAULT_CANVAS_DIR)
    canvas_dir = str(Path(canvas_dir_raw).expanduser().resolve())

    return Config(
        vault_path=vault_path,
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
        qdrant_collection=os.environ.get(
            "QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION
        ),
        ollama_host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        ollama_chat_model=os.environ.get(
            "OLLAMA_CHAT_MODEL", DEFAULT_OLLAMA_CHAT_MODEL
        ),
        ollama_embed_model=os.environ.get(
            "OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL
        ),
        canvas_dir=canvas_dir,
        embed_dim=embed_dim,
        top_k=top_k,
        max_tool_calls=max_tool_calls,
        log_level=os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL),
    )