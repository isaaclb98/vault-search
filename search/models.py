"""
search/models.py — Pydantic models for API requests/responses.

Field shapes mirror IMPLEMENTATION_DESIGN.md §"Chat pipeline" and
§"Canvas pipeline" exactly. Field names use snake_case; JSON
serialization is auto via Pydantic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------- Chat ----------------------


class ChatHistoryMessage(BaseModel):
    """A single turn in the chat history."""
    role: str = Field(..., description="user | assistant | system | tool")
    content: str = Field(..., description="Message content.")


class ChatRequest(BaseModel):
    message: str = Field(..., description="The new user message.")
    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        description="Prior turns, in order. The new message is appended.",
    )


class ChatFragment(BaseModel):
    """A fragment surfaced to the UI as a card."""
    id: str
    type: str
    preview: str = Field("", description="First ~200 chars of content.")
    score: float = 0.0


class ChatResponse(BaseModel):
    reply: str = Field("", description="Final assistant text (no tool calls).")
    fragments: list[ChatFragment] = Field(default_factory=list)
    tool_calls: int = Field(0, description="Number of query_fragments calls made.")


# ---------------------- Canvas ----------------------


class CanvasItem(BaseModel):
    ref: str = Field(..., description="Vault-relative path to the fragment.")
    order: int = Field(..., ge=0)
    x: float = Field(0, description="X position on the canvas (px).")
    y: float = Field(0, description="Y position on the canvas (px).")
    annotation: str = Field("", description="Operator note for this item.")


class CanvasSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Human-readable canvas name.")
    items: list[CanvasItem] = Field(default_factory=list)
    slug: Optional[str] = Field(
        None,
        description="Optional: existing canvas slug to update. Omit to create.",
    )


class CanvasSaveResponse(BaseModel):
    slug: str
    modified: datetime


class CanvasSummary(BaseModel):
    slug: str
    name: str
    created: datetime
    modified: datetime
    item_count: int


class CanvasListResponse(BaseModel):
    canvases: list[CanvasSummary] = Field(default_factory=list)


class CanvasDetailResponse(BaseModel):
    slug: str
    name: str
    created: datetime
    modified: datetime
    items: list[CanvasItem]


class CanvasDeleteResponse(BaseModel):
    slug: str
    deleted: bool = True


# ---------------------- Fragment fetch ----------------------


class FragmentResponse(BaseModel):
    ref: str
    content: str
    type: str
    truncated: bool = False


# ---------------------- Errors ----------------------


class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str