"""Data models for the knowledge graph."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    MEMORY = "memory"
    CHAT = "chat"
    FILE = "file"
    PERSON = "person"
    PROJECT = "project"
    IDEA = "idea"
    TAG = "tag"
    TOPIC = "topic"


class EdgeType(str, Enum):
    RELATED_TO = "related_to"
    MENTIONED_IN = "mentioned_in"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    TAGGED_WITH = "tagged_with"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"


class GraphNode(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    label: str
    description: str = ""
    metadata: dict = Field(default_factory=dict)
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    x: float = 0.0  # Layout position
    y: float = 0.0
    color: str = "#4a9eff"
    private: bool = Field(
        default=False,
        description="Created in Private mode - never written to disk",
    )


class GraphEdge(BaseModel):
    """An edge in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.RELATED_TO
    label: str = ""
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    private: bool = Field(
        default=False,
        description="Created in Private mode - never written to disk",
    )


class GraphQuery(BaseModel):
    """Query parameters for graph search."""
    text: str = ""
    node_types: list[NodeType] = Field(default_factory=list)
    edge_types: list[EdgeType] = Field(default_factory=list)
    max_depth: int = Field(default=2, ge=1, le=10)
    limit: int = Field(default=50, ge=1, le=1000)
