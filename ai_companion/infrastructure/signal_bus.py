"""Typed Qt signal bus for inter-service communication."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class ServiceSignals(QObject):
    """Common signals emitted by all services."""
    status_changed = Signal(str, str)  # service_name, status_message
    error_occurred = Signal(str, str)  # service_name, error_message


class LlmSignals(QObject):
    """Signals for the LLM service."""
    chunk_received = Signal(str, str)  # conversation_id, chunk_text
    generation_started = Signal(str)   # conversation_id
    generation_complete = Signal(str)  # conversation_id
    generation_cancelled = Signal(str) # conversation_id
    generation_error = Signal(str, str) # conversation_id, error
    model_loaded = Signal(str)         # model_name
    model_unloaded = Signal()
    model_status_changed = Signal(str, str)  # status, detail
    # conversation_id, how many earlier messages were dropped to fit n_ctx
    context_truncated = Signal(str, int)


class MemorySignals(QObject):
    """Signals for the memory service."""
    memory_added = Signal(str)          # memory_id
    memory_updated = Signal(str)        # memory_id
    memory_deleted = Signal(str)        # memory_id
    memory_approved = Signal(str)       # memory_id
    memory_rejected = Signal(str)       # memory_id
    candidate_extracted = Signal(str)   # memory_id (pending approval)


class GraphSignals(QObject):
    """Signals for the knowledge graph service."""
    node_added = Signal(str)    # node_id
    node_updated = Signal(str)  # node_id
    node_deleted = Signal(str)  # node_id
    edge_added = Signal(str)    # edge_id
    edge_deleted = Signal(str)  # edge_id
    graph_cleared = Signal()


class VaultSignals(QObject):
    """Signals for the vault service."""
    file_imported = Signal(str)     # file_path
    file_removed = Signal(str)      # file_path
    mode_changed = Signal(str)      # vault_mode
    audit_event = Signal(str, str)  # event_type, detail


class CameraSignals(QObject):
    """Signals for the camera service."""
    frame_captured = Signal(object)  # numpy array (current frame only)
    camera_opened = Signal()
    camera_closed = Signal()
    analysis_complete = Signal(str, str)  # analysis_id, result_text


class ImageGenSignals(QObject):
    """Signals for image generation."""
    generation_started = Signal(str)     # request_id
    generation_progress = Signal(str, int)  # request_id, percent
    generation_complete = Signal(str, str)  # request_id, image_path
    generation_error = Signal(str, str)     # request_id, error
    worker_status_changed = Signal(str)     # status


class SpeechSignals(QObject):
    """Signals for speech services."""
    transcription_complete = Signal(str)  # transcribed_text
    tts_started = Signal()
    tts_finished = Signal()
    tts_error = Signal(str)             # error_message
    listening_started = Signal()
    listening_stopped = Signal()
    # Hands-free voice loop.
    state_changed = Signal(str)              # off|idle|capturing|thinking|speaking
    level_changed = Signal(float)            # mic RMS, for the level meter
    transcribed = Signal(str)                # raw text heard
    prompt_ready = Signal(str, str)          # conversation_id, prompt
    # Heard clearly, but deliberately not acted on. Carries the reason so the
    # UI can explain itself instead of appearing broken.
    utterance_ignored = Signal(str, str)     # heard_text, reason


class FileWorkspaceSignals(QObject):
    """Signals for the file workspace."""
    file_operation = Signal(str, str, str)  # op_type, path, result
    audit_event = Signal(str, str, str)     # timestamp, op, detail


class CodeWorkspaceSignals(QObject):
    """Signals for the code workspace."""
    execution_started = Signal(str)       # execution_id
    execution_output = Signal(str, str)   # execution_id, output_line
    execution_complete = Signal(str, int) # execution_id, return_code
    execution_error = Signal(str, str)    # execution_id, error
    snapshot_created = Signal(str)        # snapshot_id
    snapshot_restored = Signal(str)       # snapshot_id


class Sandbox3DSignals(QObject):
    """Signals for the 3D sandbox."""
    model_imported = Signal(str)    # model_id
    model_removed = Signal(str)     # model_id
    scene_cleared = Signal()
    import_error = Signal(str, str) # file_path, error


class SignalBus(QObject):
    """Central signal bus connecting all services."""
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.service = ServiceSignals()
        self.llm = LlmSignals()
        self.memory = MemorySignals()
        self.graph = GraphSignals()
        self.vault = VaultSignals()
        self.camera = CameraSignals()
        self.image_gen = ImageGenSignals()
        self.speech = SpeechSignals()
        self.file_workspace = FileWorkspaceSignals()
        self.code_workspace = CodeWorkspaceSignals()
        self.sandbox3d = Sandbox3DSignals()
