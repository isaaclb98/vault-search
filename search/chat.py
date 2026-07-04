"""
search/chat.py — LLM chat tool loop.

The LLM has a single tool: query_fragments. This module owns the
tool-call loop:

  1. Send history + new message + tools=[query_fragments] to Ollama.
  2. If response has tool_calls, dispatch each one (call Qdrant),
     append tool results to the messages, re-send.
  3. Repeat until assistant emits a final reply (no tool calls) OR
     MAX_TOOL_CALLS is reached.

The function returns a (reply_text, fragments, tool_calls_count)
triple. The caller (route handler) maps that to the ChatResponse.

Error handling matches IMPLEMENTATION_DESIGN.md §"Error matrix":
  - OllamaUnavailable → 503 "LLM service down"
  - OllamaModelMissing → 503 "Model X not found, run `ollama pull X`"
  - QdrantUnavailable → 503 "Vector store down"
  - MAX_TOOL_CALLS exceeded → return partial response, log warning
  - LLM returns invalid JSON tool call → treat as final response
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from search import config
from search.models import ChatFragment
from search.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelMissing,
    OllamaUnavailable,
)
from search.qdrant_client import QdrantFragments, QdrantUnavailable

logger = logging.getLogger(__name__)


# Tool schema: the single tool the LLM can call. Matches
# IMPLEMENTATION_DESIGN.md §"Chat pipeline / Tool: query_fragments".
QUERY_FRAGMENTS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "query_fragments",
        "description": (
            "Find fragments in the vault semantically related to the query. "
            "Returns up to k fragments (default 8) with their type, preview, "
            "and similarity score. Optionally restrict to a single fragment type "
            "(quotation, fact, or thought)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of fragments to return.",
                    "default": 8,
                },
                "type_filter": {
                    "type": "string",
                    "enum": ["quotation", "fact", "thought"],
                    "description": "Optional fragment-type restriction.",
                },
            },
            "required": ["query"],
        },
    },
}

# How many characters of fragment content to surface as a "preview"
# in the chat card. IMPLEMENTATION_DESIGN.md §"Chat pipeline"
# specifies "first 200 chars".
PREVIEW_CHARS: int = 200

# Maximum length of a tool result message sent back to the LLM. We
# cap it so the LLM context doesn't blow up on a generous k.
TOOL_RESULT_MAX_CHARS: int = 4000


@dataclass
class ChatResult:
    reply: str
    fragments: list[ChatFragment] = field(default_factory=list)
    tool_calls: int = 0


def _system_prompt() -> str:
    """System prompt for the chat LLM. Stays under a few hundred tokens."""
    return (
        "You retrieve fragments from an Obsidian vault. Use the "
        "query_fragments tool when the user asks for fragments about a "
        "topic, source, or contrast. After receiving tool results, "
        "synthesize a brief reply that names the surfaced fragments "
        "(by their type and filename) and explains the connection. "
        "Do not invent fragment content. If the tool returns nothing "
        "useful, say so plainly."
    )


def _tool_dispatch(
    qdrant: QdrantFragments,
    ollama: OllamaClient,
    embed_dim: int,
    name: str,
    raw_args: str,
    seen_fragments: dict[str, ChatFragment],
) -> tuple[str, str]:
    """
    Dispatch a single tool call. Returns (tool_message_content, fragment_id_or_empty).

    The tool_message_content is what gets appended to the chat messages
    for the next LLM turn. The fragment_id is captured for the
    per-fragment list returned to the UI.

    Errors here raise — the caller (run_chat_loop) decides whether to
    abort the whole request or keep going.
    """
    if name != "query_fragments":
        # The LLM called a tool we didn't advertise. Return a polite
        # refusal and continue the loop (don't abort).
        return json.dumps({"error": f"unknown tool: {name}"}), ""

    # Parse arguments. The LLM is supposed to send a JSON-encoded
    # string. If it sends garbage, treat it as a final-response
    # condition by raising a ValueError that the caller catches.
    try:
        args = json.loads(raw_args) if raw_args else {}
    except (ValueError, TypeError):
        raise ValueError(f"tool {name} returned invalid JSON: {raw_args!r}")

    if not isinstance(args, dict):
        raise ValueError(f"tool {name} args must be an object, got {type(args).__name__}")

    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError(f"tool {name} requires a non-empty 'query'")
    try:
        k = int(args.get("k") or 8)
    except (TypeError, ValueError):
        k = 8
    k = max(1, min(k, 32))  # hard cap so the tool result stays small
    type_filter = args.get("type_filter")
    if type_filter not in (None, "quotation", "fact", "thought"):
        type_filter = None

    # Embed the query via Ollama. Network error → OllamaUnavailable,
    # which propagates to the route handler.
    vec = ollama.embed_one(query)
    if len(vec) != embed_dim:
        # Soft fallback: if the model returns a different dim, the
        # search will still go through but Qdrant will reject it.
        # Better to surface a clear error than to silently 500 later.
        raise OllamaError(
            f"embedding dim mismatch: model returned {len(vec)}, "
            f"EMBED_DIM={embed_dim}"
        )

    # Search Qdrant. Connection errors raise QdrantUnavailable.
    hits = qdrant.search(vec, k=k, type_filter=type_filter)

    # Build tool result (compact) and accumulate fragments for the UI.
    tool_rows: list[dict] = []
    first_id: str = ""
    for h in hits:
        if not first_id:
            first_id = h.id
        preview = h.content[:PREVIEW_CHARS]
        if h.id not in seen_fragments:
            seen_fragments[h.id] = ChatFragment(
                id=h.id,
                type=h.type,
                preview=preview,
                score=h.score,
            )
        tool_rows.append({
            "id": h.id,
            "type": h.type,
            "score": round(h.score, 4),
            "preview": preview,
        })

    result_text = json.dumps({"fragments": tool_rows}, ensure_ascii=False)
    if len(result_text) > TOOL_RESULT_MAX_CHARS:
        # Trim to the first N chars plus an explicit truncation marker.
        result_text = result_text[:TOOL_RESULT_MAX_CHARS] + "…(truncated)"
    return result_text, first_id


def run_chat_loop(
    cfg: config.Config,
    ollama: OllamaClient,
    qdrant: QdrantFragments,
    history: list[dict],
    message: str,
) -> ChatResult:
    """
    Execute the chat tool loop. Returns ChatResult.

    `history` is a list of {"role": ..., "content": ...} dicts in
    chronological order. `message` is the new user message and is
    appended last.

    Error semantics:
      - OllamaUnavailable, OllamaModelMissing, QdrantUnavailable:
        propagate. The route handler maps them to 503.
      - Other OllamaError: propagate as a 503 with the error message.
      - ValueError from invalid JSON in tool args: log, treat the
        loop as terminated, return whatever fragments we've
        accumulated + an empty reply. The route handler still
        surfaces a 200 with an empty reply — the user sees a blank
        assistant turn rather than an error page, matching the
        IMPLEMENTATION_DESIGN.md §"Error matrix" entry "LLM returns
        invalid JSON tool call: Treat as final response, skip tool
        execution".
      - MAX_TOOL_CALLS reached: log warning, return partial result.
    """
    messages: list[dict] = [{"role": "system", "content": _system_prompt()}]
    # OpenAI's chat-completions API accepts history in any order,
    # but we keep the chat messages in chronological order to make
    # debugging easier.
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    seen_fragments: dict[str, ChatFragment] = {}
    tool_call_count = 0
    final_reply = ""

    t0 = time.time()
    while tool_call_count <= cfg.max_tool_calls:
        if tool_call_count == cfg.max_tool_calls:
            # We've burned the budget. Send one more turn WITHOUT
            # tools so the LLM is forced to summarize with what it
            # has, then break.
            logger.warning(
                "MAX_TOOL_CALLS=%d reached; forcing final reply",
                cfg.max_tool_calls,
            )
            try:
                resp = ollama.chat(messages, tools=None)
            except (OllamaUnavailable, OllamaModelMissing):
                raise
            except OllamaError as e:
                logger.warning("Ollama error in forced-final turn: %s", e)
                raise OllamaUnavailable(str(e)) from e
            if resp.get("choices"):
                choice_msg = resp["choices"][0].get("message") or {}
                final_reply = choice_msg.get("content") or ""
            break

        try:
            resp = ollama.chat(messages, tools=[QUERY_FRAGMENTS_TOOL])
        except (OllamaUnavailable, OllamaModelMissing):
            raise
        except OllamaError as e:
            # Treat generic Ollama errors the same as unavailable —
            # we can't tell the user "the model is fine but the host
            # is dead" in a useful way.
            logger.warning("Ollama error in chat loop: %s", e)
            raise OllamaUnavailable(str(e)) from e

        if not resp.get("choices"):
            raise OllamaError("Ollama returned no choices")
        choice = resp["choices"][0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # Final assistant message — no tool calls.
            final_reply = msg.get("content") or ""
            break

        # Append the assistant message verbatim (including tool_calls)
        # so the next turn's history reflects what the LLM emitted.
        messages.append(msg)

        # Dispatch each tool call. We dispatch them in order; if any
        # one fails with a ValueError (invalid JSON), we record the
        # error in the tool result and continue the loop so the LLM
        # can react. Network errors propagate.
        for tc in tool_calls:
            tool_call_count += 1
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or ""
            try:
                tool_content, _ = _tool_dispatch(
                    qdrant, ollama, cfg.embed_dim, name, raw_args,
                    seen_fragments,
                )
            except ValueError as e:
                logger.warning("tool dispatch invalid args: %s", e)
                tool_content = json.dumps({"error": str(e)})
            except OllamaError:
                raise
            except QdrantUnavailable:
                raise

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or "",
                "content": tool_content,
            })

    elapsed = time.time() - t0
    logger.info(
        "chat loop done in %.2fs (tool_calls=%d, fragments=%d)",
        elapsed, tool_call_count, len(seen_fragments),
    )

    return ChatResult(
        reply=final_reply,
        fragments=list(seen_fragments.values()),
        tool_calls=tool_call_count,
    )