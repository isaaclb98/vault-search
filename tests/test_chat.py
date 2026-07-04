"""
tests/test_chat.py

Chat tool-loop coverage:
  - Single-shot final reply (no tool_calls) → fragments empty.
  - LLM emits a query_fragments call → we dispatch to Qdrant.
  - Multi-turn tool loop: LLM emits two calls then a final reply.
  - MAX_TOOL_CALLS bounds the loop and forces a final reply.
  - Invalid JSON in tool args → treated as final response.
  - Ollama unavailable / model missing / Qdrant unavailable
    propagate from run_chat_loop.
  - Fragment preview is first ~200 chars.
"""

from __future__ import annotations

import json

import pytest

from search.chat import (
    PREVIEW_CHARS,
    QUERY_FRAGMENTS_TOOL,
    run_chat_loop,
)
from search.models import ChatFragment
from search.ollama_client import OllamaModelMissing, OllamaUnavailable
from search.qdrant_client import QdrantUnavailable


# ---------------------- Helper builders ----------------------


def _tool_call_response(tool_call_id: str, name: str, arguments: str) -> dict:
    """Build an OpenAI-style chat response carrying one tool_call."""
    return {
        "id": "mock",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _final_response(content: str) -> dict:
    return {
        "id": "mock",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


# ---------------------- Final-reply-only path ----------------------


def test_chat_no_tool_calls_returns_final_reply(
    app_config, mock_ollama, qdrant_in_memory,
):
    mock_ollama.push_chat(_final_response("No fragments needed."))
    result = run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[],
        message="hi",
    )
    assert result.reply == "No fragments needed."
    assert result.tool_calls == 0
    assert result.fragments == []


# ---------------------- Tool-call dispatch ----------------------


def test_chat_tool_call_dispatches_to_qdrant(
    app_config, mock_ollama, qdrant_in_memory,
):
    # Seed Qdrant with one fragment.
    qdrant_in_memory.upsert(
        "quotations/A.md",
        [0.5] * app_config.embed_dim,
        {
            "path": "quotations/A.md",
            "content": "First 100 chars of A." + ("x" * 100),
            "type": "quotation",
            "filename": "A.md",
            "indexed_at": "2026-07-04T00:00:00Z",
        },
    )

    # LLM emits a tool call, then a final reply.
    mock_ollama.push_chat(_tool_call_response(
        "call_1", "query_fragments", json.dumps({"query": "hello", "k": 4}),
    ))
    mock_ollama.push_chat(_final_response("Here is A."))
    # Embedding response: one vector matching the seeded vector.
    mock_ollama.push_embed([[0.5] * app_config.embed_dim])

    result = run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[],
        message="hello",
    )
    assert result.reply == "Here is A."
    assert result.tool_calls == 1
    assert len(result.fragments) == 1
    assert result.fragments[0].id == "quotations/A.md"
    assert result.fragments[0].type == "quotation"


def test_chat_preview_is_first_200_chars(
    app_config, mock_ollama, qdrant_in_memory,
):
    long_content = "ABCDE" * 200  # 1000 chars
    qdrant_in_memory.upsert(
        "facts/Long.md",
        [0.1] * app_config.embed_dim,
        {
            "path": "facts/Long.md",
            "content": long_content,
            "type": "fact",
            "filename": "Long.md",
            "indexed_at": "2026-07-04T00:00:00Z",
        },
    )
    mock_ollama.push_chat(_tool_call_response(
        "call_1", "query_fragments", json.dumps({"query": "long"}),
    ))
    mock_ollama.push_chat(_final_response("ok"))
    mock_ollama.push_embed([[0.1] * app_config.embed_dim])

    result = run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[],
        message="long",
    )
    assert len(result.fragments) == 1
    assert len(result.fragments[0].preview) == PREVIEW_CHARS
    assert result.fragments[0].preview == long_content[:PREVIEW_CHARS]


def test_chat_type_filter_applied(
    app_config, mock_ollama, qdrant_in_memory,
):
    """type_filter restricts Qdrant results to one fragment type."""
    # Seed two: one quotation, one fact.
    qdrant_in_memory.upsert(
        "quotations/Q.md",
        [0.1] * app_config.embed_dim,
        {"path": "quotations/Q.md", "content": "Q", "type": "quotation",
         "filename": "Q.md", "indexed_at": "2026-07-04T00:00:00Z"},
    )
    qdrant_in_memory.upsert(
        "facts/F.md",
        [0.1] * app_config.embed_dim,
        {"path": "facts/F.md", "content": "F", "type": "fact",
         "filename": "F.md", "indexed_at": "2026-07-04T00:00:00Z"},
    )
    # LLM asks for facts only.
    mock_ollama.push_chat(_tool_call_response(
        "call_1", "query_fragments",
        json.dumps({"query": "x", "type_filter": "fact"}),
    ))
    mock_ollama.push_chat(_final_response("ok"))
    mock_ollama.push_embed([[0.1] * app_config.embed_dim])

    result = run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[],
        message="x",
    )
    assert [f.id for f in result.fragments] == ["facts/F.md"]


# ---------------------- Multi-turn loop ----------------------


def test_chat_multi_turn_tool_loop(
    app_config, mock_ollama, qdrant_in_memory,
):
    """LLM emits two tool calls, then a final reply. Both are dispatched."""
    qdrant_in_memory.upsert(
        "thoughts/A.md", [0.1] * app_config.embed_dim,
        {"path": "thoughts/A.md", "content": "A", "type": "thought",
         "filename": "A.md", "indexed_at": "2026-07-04T00:00:00Z"},
    )
    qdrant_in_memory.upsert(
        "thoughts/B.md", [0.2] * app_config.embed_dim,
        {"path": "thoughts/B.md", "content": "B", "type": "thought",
         "filename": "B.md", "indexed_at": "2026-07-04T00:00:00Z"},
    )
    mock_ollama.push_chat(_tool_call_response(
        "call_1", "query_fragments", json.dumps({"query": "alpha"}),
    ))
    mock_ollama.push_chat(_tool_call_response(
        "call_2", "query_fragments", json.dumps({"query": "beta"}),
    ))
    mock_ollama.push_chat(_final_response("Both surfaced."))
    mock_ollama.push_embed([[0.1] * app_config.embed_dim, [0.2] * app_config.embed_dim])

    result = run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[],
        message="go",
    )
    assert result.tool_calls == 2
    assert result.reply == "Both surfaced."
    assert {f.id for f in result.fragments} == {"thoughts/A.md", "thoughts/B.md"}


# ---------------------- MAX_TOOL_CALLS ----------------------


def test_chat_max_tool_calls_caps_loop(
    app_config, mock_ollama, qdrant_in_memory,
):
    """When the LLM keeps calling tools, we cap at MAX_TOOL_CALLS."""
    # Build MAX_TOOL_CALLS tool-call responses + 1 final.
    for i in range(app_config.max_tool_calls):
        mock_ollama.push_chat(_tool_call_response(
            f"call_{i}", "query_fragments",
            json.dumps({"query": f"q{i}"}),
        ))
    mock_ollama.push_chat(_final_response("Capped."))
    # Push enough embed results (one per tool call).
    for _ in range(app_config.max_tool_calls):
        mock_ollama.push_embed([[0.1] * app_config.embed_dim])

    result = run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[],
        message="loop",
    )
    # We dispatch at most MAX_TOOL_CALLS tool calls. The final
    # "no-tools" turn happens after the cap is hit (the impl issues
    # one more chat() without tools).
    assert result.tool_calls == app_config.max_tool_calls
    assert result.reply == "Capped."


# ---------------------- Invalid JSON ----------------------


def test_chat_invalid_tool_json_treated_as_final(
    app_config, mock_ollama, qdrant_in_memory, caplog,
):
    """A tool call with garbage arguments → loop terminates, no crash."""
    mock_ollama.push_chat(_tool_call_response(
        "call_1", "query_fragments", "not json",
    ))
    # The chat loop terminates after the bad tool call (the loop
    # records an error message in the tool result and stops because
    # no further chat response is queued).
    result = run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[],
        message="x",
    )
    assert result.tool_calls == 1
    assert result.fragments == []


# ---------------------- Error propagation ----------------------


def test_chat_propagates_ollama_unavailable(
    app_config, mock_ollama, qdrant_in_memory,
):
    mock_ollama.fail_next_with = OllamaUnavailable("ollama down")
    with pytest.raises(OllamaUnavailable):
        run_chat_loop(
            cfg=app_config,
            ollama=mock_ollama,
            qdrant=qdrant_in_memory,
            history=[],
            message="hi",
        )


def test_chat_propagates_ollama_model_missing(
    app_config, mock_ollama, qdrant_in_memory,
):
    mock_ollama.fail_next_with = OllamaModelMissing("model missing")
    with pytest.raises(OllamaModelMissing):
        run_chat_loop(
            cfg=app_config,
            ollama=mock_ollama,
            qdrant=qdrant_in_memory,
            history=[],
            message="hi",
        )


def test_chat_propagates_qdrant_unavailable(
    app_config, mock_ollama, qdrant_in_memory,
):
    """If the embed succeeds but Qdrant is down, the loop bubbles."""
    from search.qdrant_client import QdrantFragments

    class _BoomQdrant(QdrantFragments):
        def search(self, *a, **kw):
            raise QdrantUnavailable("qdrant down")

    mock_ollama.push_chat(_tool_call_response(
        "call_1", "query_fragments", json.dumps({"query": "x"}),
    ))
    mock_ollama.push_embed([[0.1] * app_config.embed_dim])

    boom = _BoomQdrant(
        client=qdrant_in_memory.client,
        collection=qdrant_in_memory.collection,
    )
    with pytest.raises(QdrantUnavailable):
        run_chat_loop(
            cfg=app_config,
            ollama=mock_ollama,
            qdrant=boom,
            history=[],
            message="x",
        )


# ---------------------- History plumbing ----------------------


def test_chat_history_is_forwarded(
    app_config, mock_ollama, qdrant_in_memory,
):
    mock_ollama.push_chat(_final_response("ok"))
    run_chat_loop(
        cfg=app_config,
        ollama=mock_ollama,
        qdrant=qdrant_in_memory,
        history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier reply"},
        ],
        message="now",
    )
    sent = mock_ollama.chat_calls[0]["messages"]
    # System prompt + history + new message.
    assert sent[0]["role"] == "system"
    assert sent[1]["content"] == "earlier"
    assert sent[2]["content"] == "earlier reply"
    assert sent[3]["content"] == "now"


# ---------------------- Schema sanity ----------------------


def test_query_fragments_tool_schema_is_openai_compatible():
    """The tool schema matches what /v1/chat/completions expects."""
    assert QUERY_FRAGMENTS_TOOL["type"] == "function"
    fn = QUERY_FRAGMENTS_TOOL["function"]
    assert fn["name"] == "query_fragments"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "query" in params["required"]
    assert "k" in params["properties"]
    assert "type_filter" in params["properties"]
    assert params["properties"]["type_filter"]["enum"] == ["quotation", "fact", "thought"]