"""
tests/test_indexer.py

Ingest idempotency + walk + error path coverage for the indexer.

Layering:
  - Pure-Python tests for iter_markdown_files (no I/O deps).
  - Tests using fake_vault + mock Ollama + in-memory Qdrant to
    exercise the end-to-end upsert path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from indexer.indexer import iter_markdown_files
from search.ollama_client import OllamaModelMissing, OllamaUnavailable


# ---------------------- Walk ----------------------


def test_iter_markdown_files_walks_three_dirs(fake_vault: Path):
    found = sorted(p.relative_to(fake_vault).as_posix() for _, p in iter_markdown_files(str(fake_vault)))
    assert "quotations/Calasso_Page42.md" in found
    assert "facts/Capitals_2024.md" in found
    assert "thoughts/On_ritual.md" in found
    assert "thoughts/sub/deep.md" in found


def test_iter_markdown_files_skips_other_dirs(tmp_path: Path):
    """Files outside quotations/facts/thoughts are ignored."""
    (tmp_path / "daily").mkdir()
    (tmp_path / "daily" / "should_skip.md").write_text("x", encoding="utf-8")
    (tmp_path / "quotations").mkdir()
    (tmp_path / "quotations" / "k.md").write_text("x", encoding="utf-8")
    found = [p.name for _, p in iter_markdown_files(str(tmp_path))]
    assert found == ["k.md"]


def test_iter_markdown_files_missing_vault(tmp_path: Path):
    assert list(iter_markdown_files(str(tmp_path / "nope"))) == []


def test_iter_markdown_files_empty_vault(tmp_path: Path):
    (tmp_path / "quotations").mkdir()
    assert list(iter_markdown_files(str(tmp_path))) == []


def test_iter_markdown_files_yields_type(fake_vault: Path):
    """Type is the first path component (quotation/fact/thought)."""
    types = {t for t, _ in iter_markdown_files(str(fake_vault))}
    assert types == {"quotations", "facts", "thoughts"}


# ---------------------- End-to-end indexer ----------------------


@pytest.fixture
def _patched_qdrant_client(monkeypatch, qdrant_in_memory, mock_ollama):
    """
    Patch QdrantClient + OllamaClient in indexer.indexer so the
    indexer uses our in-memory Qdrant and our mock Ollama instead
    of building real network clients.
    """
    from qdrant_client import QdrantClient
    monkeypatch.setattr(
        "indexer.indexer.QdrantClient",
        lambda **kwargs: qdrant_in_memory.client,
    )
    monkeypatch.setattr(
        "indexer.indexer.OllamaClient",
        lambda **kwargs: mock_ollama,
    )


def test_indexer_end_to_end(
    fake_vault: Path,
    monkeypatch,
    mock_ollama,
    app_config,
    _patched_qdrant_client,
    qdrant_in_memory,
):
    """Walk vault, embed, upsert; assertions on Qdrant state."""
    from indexer import indexer as indexer_mod

    monkeypatch.setenv("VAULT_PATH", str(fake_vault))

    # Three real fragments + one empty file skipped = 4 indexes.
    mock_ollama.push_embed([
        [0.1] * 768,
        [0.2] * 768,
        [0.3] * 768,
        [0.4] * 768,
    ])
    # Push another batch in case the indexer loops more than once.
    mock_ollama.push_embed([[0.5] * 768])

    rc = indexer_mod.main([])
    assert rc == 0

    # All 4 non-empty fragments should be present.
    hits = qdrant_in_memory.search([0.1] * 768, k=10)
    paths = sorted(h.path for h in hits)
    assert "quotations/Calasso_Page42.md" in paths
    assert "facts/Capitals_2024.md" in paths
    assert "thoughts/On_ritual.md" in paths
    assert "thoughts/sub/deep.md" in paths


def test_indexer_is_idempotent(
    fake_vault: Path,
    monkeypatch,
    mock_ollama,
    app_config,
    _patched_qdrant_client,
    qdrant_in_memory,
):
    """Re-running upserts the same points (no duplicate row ids)."""
    from indexer import indexer as indexer_mod

    monkeypatch.setenv("VAULT_PATH", str(fake_vault))

    # Two runs, each consuming 4 vectors.
    mock_ollama.push_embed([[0.1] * 768] * 4)
    mock_ollama.push_embed([[0.1] * 768] * 4)

    assert indexer_mod.main([]) == 0
    assert indexer_mod.main([]) == 0

    # Total points still 4 — no duplicates from re-upsert.
    hits = qdrant_in_memory.search([0.1] * 768, k=100)
    assert len(hits) == 4


def test_indexer_empty_vault_returns_zero(
    tmp_path: Path,
    monkeypatch,
    mock_ollama,
    app_config,
    _patched_qdrant_client,
    capsys,
):
    """No .md files anywhere → exit 0 with a warning on stderr."""
    from indexer import indexer as indexer_mod

    empty = tmp_path / "empty_vault"
    empty.mkdir()
    monkeypatch.setenv("VAULT_PATH", str(empty))

    rc = indexer_mod.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "no .md files" in captured.err


def test_indexer_vault_path_missing_returns_1(
    tmp_path: Path,
    monkeypatch,
    app_config,
    _patched_qdrant_client,
):
    from indexer import indexer as indexer_mod

    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "does_not_exist"))
    rc = indexer_mod.main([])
    assert rc == 1


def test_indexer_embed_model_missing_returns_3(
    fake_vault: Path,
    monkeypatch,
    mock_ollama,
    app_config,
    _patched_qdrant_client,
):
    """Ollama 404 on the embed probe → exit 3."""
    from indexer import indexer as indexer_mod

    monkeypatch.setenv("VAULT_PATH", str(fake_vault))
    mock_ollama.fail_next_with = OllamaModelMissing(
        "model nomic-embed-text not loaded"
    )
    rc = indexer_mod.main([])
    assert rc == 3


def test_indexer_ollama_unreachable_returns_3(
    fake_vault: Path,
    monkeypatch,
    mock_ollama,
    app_config,
    _patched_qdrant_client,
):
    """Ollama healthz fails → exit 3."""
    from indexer import indexer as indexer_mod

    monkeypatch.setenv("VAULT_PATH", str(fake_vault))
    mock_ollama.healthz_result = False
    rc = indexer_mod.main([])
    assert rc == 3


def test_indexer_qdrant_unreachable_returns_2(
    fake_vault: Path,
    monkeypatch,
    mock_ollama,
    app_config,
):
    """Qdrant connect failure → exit 2."""
    from indexer import indexer as indexer_mod
    from qdrant_client.http.exceptions import UnexpectedResponse

    def boom(**kwargs):
        raise UnexpectedResponse("nope", 503, b"", b"")

    monkeypatch.setattr("indexer.indexer.QdrantClient", boom)
    monkeypatch.setenv("VAULT_PATH", str(fake_vault))
    rc = indexer_mod.main([])
    assert rc == 2