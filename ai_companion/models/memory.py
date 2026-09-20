"""Data models for the memory system."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class MemorySource(str, Enum):
    USER_EXPLICIT = "user_explicit"
    EXTRACTED = "extracted"
    IMPORTED = "imported"


class Memory(BaseModel):
    """A single memory with provenance and confidence."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: MemorySource = MemorySource.USER_EXPLICIT
    provenance: str = ""  # Where this memory came from
    status: MemoryStatus = MemoryStatus.PENDING
    pinned: bool = False
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    modified: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    related_entity_ids: list[str] = Field(default_factory=list)
    source_conversation_id: Optional[str] = None
    source_message_id: Optional[str] = None
    private: bool = Field(
        default=False,
        description="Created in Private mode - never written to disk",
    )


class MemoryFilter(BaseModel):
    """Filter criteria for memory search."""
    query: str = ""
    tags: list[str] = Field(default_factory=list)
    status: Optional[MemoryStatus] = None
    pinned: Optional[bool] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
