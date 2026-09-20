"""Pydantic models for configuration."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, Field


class LlmConfig(BaseModel):
    """Configuration for the LLM backend."""
    model_path: str = Field(default="", description="Path to GGUF model file")
    context_length: int = Field(default=4096, ge=512, le=131072)
    gpu_layers: int = Field(default=0, ge=0, le=100)
    threads: int = Field(default=4, ge=1, le=64)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    host: str = Field(default="127.0.0.1", description="Must be loopback")
    conversations_path: str = Field(
        default="data/conversations.json",
        description="Where chat history is persisted",
    )
    max_stored_conversations: int = Field(default=500, ge=1, le=100000)
    system_prompt: str = Field(
        default=(
            "You are a helpful local AI assistant. Answer directly and "
            "concisely. Do not narrate, explain, or evaluate your own "
            "responses. Do not add meta-commentary about the answer. "
            "Use lists only when the user asks for a list."
        ),
        max_length=4000,
        description="System prompt used for new conversations",
    )


class VaultMode(str, Enum):
    """Vault operating modes."""
    NORMAL = "normal"
    PRIVATE = "private"


class VaultConfig(BaseModel):
    """Configuration for the Private Vault."""

    # Default NORMAL, deliberately changed from PRIVATE.
    #
    # PRIVATE was the fail-safe choice: write nothing until the owner opts in.
    # In practice it meant a fresh install silently discarded every
    # conversation, and the user only discovered it after losing work. A
    # privacy default that surprises the owner is a usability bug even when it
    # is technically the safer value. Private is now one click away in the
    # chat header instead of being the unannounced default.
    mode: VaultMode = VaultMode.NORMAL
    vault_root: str = "vault"
    approved_folders: list[str] = Field(default_factory=list)
    max_file_size_mb: int = Field(default=100, ge=1, le=10000)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".txt", ".md", ".json", ".csv", ".png", ".jpg", ".jpeg",
            ".gif", ".webp", ".pdf", ".docx", ".xlsx", ".py", ".js",
            ".html", ".css", ".glb", ".gltf", ".obj", ".wav", ".mp3",
        ]
    )


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""
    store_path: str = "data/memories.json"
    max_memories: int = Field(default=10000, ge=100, le=100000)
    auto_extract: bool = False
    require_approval: bool = True


class MindMapConfig(BaseModel):
    """Configuration for the Mind Map."""

    store_path: str = "data/mind_map.json"
    # Overviews are LLM-written and expensive on CPU, so they are generated
    # on demand rather than in the background.
    auto_synthesise: bool = False
    inject_overviews: bool = True
    max_injected: int = Field(default=3, ge=0, le=12)


class GraphConfig(BaseModel):
    """Configuration for the knowledge graph."""
    store_path: str = "data/graph.json"
    max_nodes: int = Field(default=5000, ge=100, le=100000)
    max_edges: int = Field(default=20000, ge=100, le=500000)


class CameraConfig(BaseModel):
    """Configuration for camera access."""
    device_index: int = Field(default=0, ge=0, le=10)
    resolution_width: int = Field(default=640, ge=160, le=3840)
    resolution_height: int = Field(default=480, ge=120, le=2160)
    auto_record: bool = Field(default=False, description="Must always be False")


class ImageGenConfig(BaseModel):
    """Configuration for image generation."""
    comfyui_path: str = ""
    host: str = Field(default="127.0.0.1", description="Must be loopback")
    port: int = Field(default=8188, ge=1024, le=65535)
    default_width: int = Field(default=512, ge=64, le=2048)
    default_height: int = Field(default=512, ge=64, le=2048)
    output_dir: str = "generated_images"


class SpeechConfig(BaseModel):
    """Configuration for speech services."""

    # base.en is the recommended default: ~10x realtime on CPU with int8,
    # and noticeably more accurate than tiny on conversational speech.
    whisper_model: str = "base.en"
    tts_engine: str = "piper"
    tts_rate: int = Field(default=150, ge=50, le=300)
    input_device_index: Optional[int] = None

    # Piper voice, without the .onnx extension.
    piper_voice: str = ""
    voices_dir: str = "models/piper"

    # Hands-free loop.
    wake_word: str = "jarvis"
    require_wake_word: bool = True
    # Mic RMS below this counts as silence. Raise it in a noisy room.
    silence_threshold: float = Field(default=0.012, ge=0.001, le=0.2)
    # Quiet time before an utterance is considered finished.
    silence_seconds: float = Field(default=1.0, ge=0.3, le=5.0)
    # Speak replies only when the question arrived by voice.
    speak_replies: bool = True
    # Post-processing preset applied to synthesised speech. Piper's voice is
    # already close in timbre; what makes it sound like a room AI is the
    # production - EQ, a short room reverb, slight detune, even level.
    voice_fx: str = "natural"

    # Which synthesiser speaks. piper is ~10x realtime and slightly flatter;
    # kokoro is ~1.2x realtime and generally judged more natural. Speed vs
    # quality, and only you can decide which you prefer.
    tts_engine_name: str = "piper"
    kokoro_voice: str = "bm_george"
    kokoro_dir: str = "models/kokoro"

    # After a reply, accept a follow-up without repeating the wake word.
    # Requiring it every sentence makes conversation feel like dictation.
    follow_up_seconds: float = Field(default=12.0, ge=0.0, le=60.0)


class CodeWorkspaceConfig(BaseModel):
    """Configuration for the code workspace."""
    sandbox_dir: str = "sandbox"
    snapshot_dir: str = "sandbox/.snapshots"
    max_execution_time: int = Field(default=30, ge=1, le=300)
    max_memory_mb: int = Field(default=512, ge=64, le=4096)
    allow_network: bool = Field(default=False, description="Must always be False")


class Sandbox3DConfig(BaseModel):
    """Configuration for the 3D sandbox."""
    max_import_size_mb: int = Field(default=50, ge=1, le=500)
    supported_formats: list[str] = Field(
        default_factory=lambda: [".glb", ".gltf", ".obj"]
    )


class AppConfig(BaseModel):
    """Root application configuration.

    Every filesystem path the app writes to is registered in PATH_FIELDS.
    `relocate()` rewrites all of them under a single root, which is how tests
    guarantee isolation from real user data. A test introspects every config
    model and fails if a new path-like field is added without registering it,
    so this cannot silently rot.
    """

    # (section, field_name, path relative to the relocation root)
    # section None means the field lives on AppConfig itself.
    PATH_FIELDS: ClassVar[tuple[tuple[Optional[str], str, str], ...]] = (
        (None, "data_dir", "data"),
        ("llm", "conversations_path", "data/conversations.json"),
        ("memory", "store_path", "data/memories.json"),
        ("graph", "store_path", "data/graph.json"),
        ("mind_map", "store_path", "data/mind_map.json"),
        ("vault", "vault_root", "vault"),
        ("code_workspace", "sandbox_dir", "sandbox"),
        ("code_workspace", "snapshot_dir", "sandbox/.snapshots"),
        ("image_gen", "output_dir", "generated_images"),
        ("speech", "voices_dir", "models/piper"),
        ("speech", "kokoro_dir", "models/kokoro"),
    )

    # Path-like fields that are deliberately NOT relocated, with the reason.
    # Listed so the coverage test can distinguish "exempt" from "forgotten".
    PATH_FIELDS_EXEMPT: ClassVar[dict[str, str]] = {
        "llm.model_path": "user-chosen GGUF, lives outside the app",
        "image_gen.comfyui_path": "user-chosen external install",
        "vault.approved_folders": "user-granted folders, must not be rewritten",
    }

    version: str = "1"
    data_dir: str = "data"
    llm: LlmConfig = Field(default_factory=LlmConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    mind_map: MindMapConfig = Field(default_factory=MindMapConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    code_workspace: CodeWorkspaceConfig = Field(default_factory=CodeWorkspaceConfig)
    sandbox3d: Sandbox3DConfig = Field(default_factory=Sandbox3DConfig)

    def relocate(self, root: str | Path) -> "AppConfig":
        """Rewrite every registered store path to live under `root`.

        Used by tests to guarantee no test touches real user data, and
        available at runtime for a portable//relocated install.
        Returns self for chaining.
        """
        base = Path(root)
        for section, field, relative in self.PATH_FIELDS:
            target = str(base / relative)
            obj = self if section is None else getattr(self, section)
            setattr(obj, field, target)
        return self
