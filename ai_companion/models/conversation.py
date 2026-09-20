"""Data models for conversations and messages."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Attachment(BaseModel):
    """An attachment to a message."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    mime_type: str = "application/octet-stream"
    size_bytes: int = 0


class Message(BaseModel):
    """A single message in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    attachments: list[Attachment] = Field(default_factory=list)
    tokens_used: int = 0
    is_streaming: bool = False
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Non-displayed extras. 'prompt_content' holds the "
            "attachment-expanded text sent to the model, so the transcript "
            "can show what the user typed while the model sees the file."
        ),
    )


class Conversation(BaseModel):
    """A conversation with message history."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Conversation"
    messages: list[Message] = Field(default_factory=list)
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    modified: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    system_prompt: str = "You are a helpful AI assistant."
    model_name: str = ""
    private: bool = Field(
        default=False,
        description="Created in Private mode — never written to disk",
    )

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        """Add a message and return it."""
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.modified = datetime.now(timezone.utc).isoformat()
        return msg

    def get_history(self, max_messages: int = 50) -> list[dict[str, str]]:
        """Get message history formatted for LLM context.

        Uses `metadata['prompt_content']` when present so attached file text
        reaches the model, while `content` stays what the user actually typed.
        """
        msgs = self.messages[-max_messages:]
        return [
            {
                "role": m.role.value,
                "content": m.metadata.get("prompt_content") or m.content,
            }
            for m in msgs
        ]
