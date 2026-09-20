"""LLM service wrapping llama-cpp-python for local inference."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional

from PySide6.QtCore import QTimer

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.privacy import PrivacyMixin
from ai_companion.models.config_models import LlmConfig
from ai_companion.models.conversation import Conversation, MessageRole


class LlmService(PrivacyMixin, BaseService):
    """Manages local LLM inference via llama-cpp-python.

    Provides streaming generation with cancellation support.
    All inference is local — no network calls.
    """

    service_name = "LLM"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model: Any = None
        self._model_loaded = False
        self._generating = False
        self._cancel_requested = False
        self._current_thread: Optional[threading.Thread] = None
        self._conversations: dict[str, Conversation] = {}
        self._current_model_name: str = ""
        self._store: Optional[JsonStore] = None
        self._store_lock = threading.Lock()
        # The conversation the user is currently talking in. Only this one is
        # ambiguous when the vault mode changes mid-session.
        self._active_conversation_id: str = ""

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def is_generating(self) -> bool:
        return self._generating

    @property
    def model_name(self) -> str:
        return self._current_model_name

    def start(self) -> None:
        super().start()
        self._signal_bus.vault.mode_changed.connect(self.on_vault_mode_changed)
        self._load_conversations()
        llm_config = self._config.llm
        if llm_config.model_path:
            try:
                self.load_model(llm_config.model_path)
            except Exception as e:
                self.emit_error(f"Failed to load model: {e}")

    def stop(self) -> None:
        # Disconnect from the bus FIRST. A stopped service that is still
        # subscribed will keep reacting to vault mode changes and rewrite the
        # store out from under the live service.
        try:
            self._signal_bus.vault.mode_changed.disconnect(
                self.on_vault_mode_changed
            )
        except (RuntimeError, TypeError):
            pass  # not connected

        self.cancel_generation()
        # Give an in-flight generation a moment to finish writing its message
        # so we do not persist a half-saved conversation.
        thread = self._current_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._save_conversations()
        self._unload_model()
        super().stop()

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------

    def on_vault_mode_changed(self, mode: str) -> None:
        """React to a Vault mode switch.

        Entering PRIVATE marks only the ACTIVE conversation private and scrubs
        its already-persisted half from disk, so a chat continued in Private
        mode does not leave its earlier turns readable. It stays usable in
        memory.

        Conversations merely restored from disk are NOT touched. An earlier
        version flagged every loaded conversation, which silently deleted the
        user's entire saved chat history the moment they toggled Private —
        data loss, not privacy. Old conversations are finished records, the
        same as memories and graph nodes; only the live session is ambiguous.
        """
        if mode != "private":
            self.emit_status("Normal mode: new chats will be saved")
            return

        active = self._active_conversation_id
        scrubbed = 0
        if active and active in self._conversations:
            conv = self._conversations[active]
            if not conv.private:
                conv.private = True
                scrubbed = 1

        if scrubbed:
            self._save_conversations()
            self.emit_status(
                "Private mode: active chat removed from disk and will not be saved"
            )
        else:
            self.emit_status("Private mode: new chats will not be saved")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_store(self) -> JsonStore:
        if self._store is None:
            self._store = JsonStore(self._config.llm.conversations_path)
        return self._store

    def _load_conversations(self) -> None:
        """Load saved conversations from disk.

        A corrupt or unreadable store must never prevent the app from
        starting — we log, keep an empty set, and carry on.
        """
        try:
            store = self._get_store()
            store.load()
            loaded: dict[str, Conversation] = {}
            for key, raw in store.items():
                try:
                    conv = Conversation.model_validate(raw)
                except Exception as exc:  # noqa: BLE001
                    self.emit_error(
                        f"Skipped unreadable conversation {key}: {exc}"
                    )
                    continue
                loaded[conv.id] = conv
            self._conversations = loaded
            if loaded:
                self.emit_status(f"Loaded {len(loaded)} conversation(s)")
        except Exception as exc:  # noqa: BLE001
            self.emit_error(f"Could not load conversation history: {exc}")
            self._conversations = {}

    def _save_conversations(self) -> None:
        """Persist all conversations atomically.

        Empty conversations are not written — they are noise from clicking
        New. Oldest are trimmed past max_stored_conversations.
        """
        try:
            with self._store_lock:
                store = self._get_store()
                store.load()
                store.clear()

                keep = [
                    c
                    for c in self._conversations.values()
                    if c.messages and not c.private
                ]
                keep.sort(key=lambda c: c.modified, reverse=True)
                limit = self._config.llm.max_stored_conversations
                for conv in keep[:limit]:
                    store.set(conv.id, conv.model_dump(mode="json"))
                store.save()
        except Exception as exc:  # noqa: BLE001
            self.emit_error(f"Could not save conversation history: {exc}")

    def load_model(self, model_path: str) -> None:
        """Load a GGUF model file."""
        if self._model_loaded:
            self._unload_model()

        self._signal_bus.llm.model_status_changed.emit("loading", model_path)
        self.emit_status(f"Loading model: {model_path}")

        # Check the file before handing it to llama.cpp, which reports a
        # missing or unreadable file as an opaque low-level error.
        from pathlib import Path as _Path

        candidate = _Path(model_path)
        if not candidate.exists():
            msg = f"Model file not found: {model_path}"
            self.emit_error(msg)
            self._signal_bus.llm.model_status_changed.emit("error", msg)
            raise FileNotFoundError(msg)
        if not candidate.is_file():
            msg = f"Model path is not a file: {model_path}"
            self.emit_error(msg)
            self._signal_bus.llm.model_status_changed.emit("error", msg)
            raise IsADirectoryError(msg)
        if candidate.suffix.lower() != ".gguf":
            msg = f"Model must be a .gguf file, got '{candidate.suffix}'"
            self.emit_error(msg)
            self._signal_bus.llm.model_status_changed.emit("error", msg)
            raise ValueError(msg)

        try:
            from llama_cpp import Llama

            llm_config = self._config.llm
            self._model = Llama(
                model_path=model_path,
                n_ctx=llm_config.context_length,
                n_gpu_layers=llm_config.gpu_layers,
                n_threads=llm_config.threads,
                verbose=False,
            )
            self._model_loaded = True
            # Extract just the filename for display
            from pathlib import Path
            self._current_model_name = Path(model_path).stem
            self._signal_bus.llm.model_loaded.emit(self._current_model_name)
            self._signal_bus.llm.model_status_changed.emit(
                "loaded", self._current_model_name
            )
            self.emit_status(f"Model loaded: {self._current_model_name}")
        except ImportError as exc:
            msg = (
                "llama-cpp-python is not installed in this environment. "
                "Install it with:\n"
                "  pip install llama-cpp-python "
                "--extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )
            self._logger.exception("llama_cpp import failed")
            self.emit_error(msg)
            self._signal_bus.llm.model_status_changed.emit("error", msg)
            raise RuntimeError(msg) from exc
        except Exception as e:
            # Log the full traceback: the surfaced message is often a terse
            # wrapper around the real cause.
            self._logger.exception("Model load failed for %s", model_path)
            self.emit_error(f"Model load failed: {e}")
            self._signal_bus.llm.model_status_changed.emit("error", str(e))
            raise

    def _unload_model(self) -> None:
        """Unload the current model."""
        self._model = None
        self._model_loaded = False
        self._current_model_name = ""
        self._signal_bus.llm.model_unloaded.emit()
        self._signal_bus.llm.model_status_changed.emit("unloaded", "")

    def _maybe_extract_memories(self, conversation_id: str, text: str) -> None:
        """Offer memory candidates from what the user just typed.

        Opt-in via Settings -> Memory. Off by default because an extractor
        that silently fills the review queue is something the owner should
        choose, not inherit.

        Only the USER's words are scanned. Extracting from model output would
        let a hallucination become a stored fact about the owner.
        """
        if not getattr(self._config.memory, "auto_extract", False):
            return

        manager = getattr(self, "_service_manager", None)
        if manager is None:
            return
        try:
            memory_service = manager.get("Memory")
        except Exception:  # noqa: BLE001
            return
        if memory_service is None:
            return

        try:
            created = memory_service.extract_candidates(
                text, conversation_id=conversation_id
            )
        except Exception as exc:  # noqa: BLE001 - never block the send
            self._logger.warning("Memory extraction skipped: %s", exc)
            return

        if created:
            plural = "y" if len(created) == 1 else "ies"
            self.emit_status(
                f"{len(created)} memor{plural} proposed - review in the MEM "
                "tab"
            )

    def _compose_system_prompt(self, base_prompt: str, query: str = "") -> str:
        """Fold approved memories into the system prompt.

        Private mode is respected implicitly: memories created privately were
        never written to disk, but they DO live in the running service, and a
        private chat is still the user's own session - so in-session memories
        remain available. Nothing new is persisted either way.
        """
        from ai_companion.services.memory_injection import compose_system_prompt

        memory_service = None
        manager = getattr(self, "_service_manager", None)
        if manager is not None:
            try:
                memory_service = manager.get("Memory")
            except Exception:  # noqa: BLE001 - memory must never break chat
                memory_service = None

        if memory_service is None:
            return (base_prompt or "").strip()

        try:
            memories = memory_service.search()
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the send
            self._logger.warning("Memory injection skipped: %s", exc)
            return (base_prompt or "").strip()

        # Widen the query with sibling tags from the graph, so asking about
        # "discord" can also surface memories tagged only "bots".
        expanded = query
        if query and manager is not None:
            try:
                from ai_companion.services.memory_relevance import (
                    expand_query_with_graph,
                )

                expanded = expand_query_with_graph(query, manager.get("Graph"))
            except Exception:  # noqa: BLE001 - enhancement only
                expanded = query

        # Mind Map overviews sit alongside raw memories, never instead of
        # them - the map assists recall, it does not replace it.
        overviews: list = []
        if manager is not None:
            try:
                mind_map = manager.get("MindMap")
                if mind_map is not None:
                    overviews = mind_map.overviews_for_query(query)
            except Exception:  # noqa: BLE001 - enhancement only
                overviews = []

        return compose_system_prompt(
            base_prompt, memories, query=expanded, overviews=overviews
        )

    def create_conversation(self, system_prompt: str = "") -> str:
        """Create a new conversation and return its ID."""
        default_prompt = (
            self._config.llm.system_prompt.strip()
            or "You are a helpful AI assistant."
        )
        conv = Conversation(
            system_prompt=system_prompt or default_prompt,
            model_name=self._current_model_name,
            private=self.is_private_mode(),
        )
        self._conversations[conv.id] = conv
        self._active_conversation_id = conv.id
        return conv.id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    def list_conversations(self) -> list[Conversation]:
        return list(self._conversations.values())

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self._save_conversations()
            return True
        return False

    def send_message(
        self,
        conversation_id: str,
        content: str,
        attachments: Optional[list] = None,
    ) -> None:
        """Send a user message and start streaming generation.

        This runs inference in a background thread to keep the UI responsive.
        """
        conv = self._conversations.get(conversation_id)
        if not conv:
            self.emit_error(f"Conversation {conversation_id} not found")
            return

        if not self._model_loaded:
            self.emit_error("No model loaded")
            return

        if self._generating:
            self.emit_error("Generation already in progress")
            return

        self._active_conversation_id = conversation_id

        # Extract text from attachments and prepend it to the prompt.
        # The model is text-only, so unreadable files produce an explicit
        # note rather than silently vanishing.
        prompt_content = content
        if attachments:
            from ai_companion.services.attachment_reader import (
                build_prompt_prefix,
                extract,
            )

            results = []
            for item in attachments:
                path = item if isinstance(item, str) else getattr(
                    item, "file_path", ""
                )
                if not path:
                    continue
                result = extract(path)
                results.append(result)
                if result.ok:
                    self.emit_status(
                        f"Attached {result.filename} "
                        f"({result.char_count:,} chars)"
                    )
                else:
                    self.emit_status(f"{result.filename}: {result.note}")
            prefix = build_prompt_prefix(results)
            if prefix:
                prompt_content = prefix + content

        # Add user message. Store the visible text, but send the
        # attachment-expanded version to the model.
        conv.add_message(MessageRole.USER, content)
        self._maybe_extract_memories(conversation_id, content)
        if prompt_content != content:
            conv.messages[-1].metadata["prompt_content"] = prompt_content

        # Start generation in background thread
        self._cancel_requested = False
        self._generating = True
        self._current_thread = threading.Thread(
            target=self._generate_worker,
            args=(conversation_id,),
            daemon=True,
        )
        self._current_thread.start()

    def cancel_generation(self) -> None:
        """Request cancellation of current generation."""
        if self._generating:
            self._cancel_requested = True

    def _generate_worker(self, conversation_id: str) -> None:
        """Background worker for text generation."""
        conv = self._conversations[conversation_id]
        llm_config = self._config.llm

        try:
            self._signal_bus.llm.generation_started.emit(conversation_id)

            # Build messages for the LLM. Approved memories are folded into
            # the system prompt at send time (not at conversation creation)
            # so edits in the Memory tab take effect on the very next message
            # instead of only in new conversations.
            messages = []
            latest_user = ""
            for message in reversed(conv.messages):
                if message.role == MessageRole.USER:
                    latest_user = message.content
                    break
            system_content = self._compose_system_prompt(
                conv.system_prompt, latest_user
            )
            if system_content:
                messages.append({
                    "role": "system",
                    "content": system_content,
                })
            messages.extend(conv.get_history())

            # Fit prompt + reply inside n_ctx. Without this, memories plus an
            # attachment plus a long conversation silently exceed the window
            # and llama.cpp raises a raw C-level error after the user has
            # already waited.
            from ai_companion.services.context_budget import fit_to_context

            budget = fit_to_context(
                messages,
                llm_config.context_length,
                llm_config.max_tokens,
            )
            messages = budget.messages
            for note in budget.notes:
                self.emit_status(note)
            if budget.truncated:
                self._signal_bus.llm.context_truncated.emit(
                    conversation_id, budget.dropped_messages
                )

            # Create assistant message placeholder
            assistant_content = ""

            # Stream tokens
            stream = self._model.create_chat_completion(
                messages=messages,
                max_tokens=budget.max_tokens,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                stream=True,
            )

            for chunk in stream:
                if self._cancel_requested:
                    break

                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    assistant_content += token
                    self._signal_bus.llm.chunk_received.emit(
                        conversation_id, token
                    )

            # Save the complete message
            if self._cancel_requested:
                if assistant_content:
                    conv.add_message(
                        MessageRole.ASSISTANT, assistant_content + " [cancelled]"
                    )
                self._signal_bus.llm.generation_cancelled.emit(conversation_id)
            else:
                conv.add_message(MessageRole.ASSISTANT, assistant_content)
                self._signal_bus.llm.generation_complete.emit(conversation_id)

            # Persist after every exchange, not just at shutdown — a crash or
            # a force-quit must not lose the conversation.
            self._autotitle(conv)
            self._save_conversations()

        except Exception as e:
            self._signal_bus.llm.generation_error.emit(conversation_id, str(e))
            self.emit_error(f"Generation error: {e}")
        finally:
            self._generating = False
            self._cancel_requested = False

    @staticmethod
    def _autotitle(conv: Conversation) -> None:
        """Name a conversation after its first user message."""
        if conv.title and conv.title != "New Conversation":
            return
        for msg in conv.messages:
            if msg.role == MessageRole.USER and msg.content.strip():
                text = " ".join(msg.content.split())
                conv.title = text[:48] + ("..." if len(text) > 48 else "")
                return

    def get_status(self) -> dict:
        """Get current LLM service status."""
        return {
            "model_loaded": self._model_loaded,
            "model_name": self._current_model_name,
            "generating": self._generating,
            "conversation_count": len(self._conversations),
        }
