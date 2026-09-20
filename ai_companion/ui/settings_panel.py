"""Settings dialog."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class SettingsPanel(QDialog):
    """Application settings dialog."""

    def __init__(
        self, service_manager: ServiceManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._sm = service_manager
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 500)
        self._initial_model_path = service_manager.config.llm.model_path.strip()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # LLM Settings
        llm_tab = QWidget()
        llm_form = QFormLayout(llm_tab)

        self._model_path = QLineEdit(self._sm.config.llm.model_path)
        model_row = QHBoxLayout()
        model_row.addWidget(self._model_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(browse_btn)
        llm_form.addRow("Model Path:", model_row)

        self._ctx_length = QSpinBox()
        self._ctx_length.setRange(512, 131072)
        self._ctx_length.setValue(self._sm.config.llm.context_length)
        llm_form.addRow("Context Length:", self._ctx_length)

        self._gpu_layers = QSpinBox()
        self._gpu_layers.setRange(0, 100)
        self._gpu_layers.setValue(self._sm.config.llm.gpu_layers)
        llm_form.addRow("GPU Layers:", self._gpu_layers)

        self._threads = QSpinBox()
        self._threads.setRange(1, 64)
        self._threads.setValue(self._sm.config.llm.threads)
        llm_form.addRow("Threads:", self._threads)

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.0, 2.0)
        self._temperature.setSingleStep(0.1)
        self._temperature.setValue(self._sm.config.llm.temperature)
        llm_form.addRow("Temperature:", self._temperature)

        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(1, 32768)
        self._max_tokens.setValue(self._sm.config.llm.max_tokens)
        llm_form.addRow("Max Tokens:", self._max_tokens)

        self._system_prompt = QTextEdit()
        self._system_prompt.setPlainText(self._sm.config.llm.system_prompt)
        self._system_prompt.setAcceptRichText(False)
        self._system_prompt.setMinimumHeight(90)
        self._system_prompt.setToolTip(
            "Applied to NEW conversations. Existing chats keep their prompt."
        )
        llm_form.addRow("System Prompt:", self._system_prompt)

        tabs.addTab(llm_tab, "LLM")

        # Vault Settings
        vault_tab = QWidget()
        vault_form = QFormLayout(vault_tab)

        # Vault mode is ALSO controlled by the chat-header badge. This dialog
        # captures its value when it opens, so writing it back unconditionally
        # on Save silently reverts a badge click made while the dialog was
        # open. Only apply it if the user actually changed it here.
        self._vault_mode = QComboBox()
        self._vault_mode.addItems(["private", "normal"])
        self._vault_mode.setCurrentText(self._sm.config.vault.mode.value)
        self._initial_vault_mode = self._vault_mode.currentText()
        vault_form.addRow("Vault Mode:", self._vault_mode)

        self._max_file_size = QSpinBox()
        self._max_file_size.setRange(1, 10000)
        self._max_file_size.setValue(self._sm.config.vault.max_file_size_mb)
        self._max_file_size.setSuffix(" MB")
        vault_form.addRow("Max File Size:", self._max_file_size)

        tabs.addTab(vault_tab, "Vault")

        # Memory Settings
        memory_tab = QWidget()
        memory_form = QFormLayout(memory_tab)

        self._auto_extract = QCheckBox(
            "Propose memories from what I type"
        )
        self._auto_extract.setChecked(
            bool(getattr(self._sm.config.memory, "auto_extract", False))
        )
        self._auto_extract.setToolTip(
            "Scans your messages for durable facts and adds them to the MEM "
            "tab as PENDING. Nothing reaches the model until you approve it.\n"
            "Disabled in private mode."
        )
        memory_form.addRow("Auto-extract:", self._auto_extract)

        extract_note = QLabel(
            "Candidates are proposals, never facts. They stay pending until "
            "you approve them in the MEM tab, and are never extracted in "
            "private mode."
        )
        extract_note.setObjectName("MetaText")
        extract_note.setWordWrap(True)
        memory_form.addRow("", extract_note)

        tabs.addTab(memory_tab, "Memory")

        # Speech Settings
        speech_tab = QWidget()
        speech_form = QFormLayout(speech_tab)

        # Microphone picker. Without this the app uses the OS default, which
        # on Windows is frequently the wrong device (a webcam mic, or a
        # disconnected headset) and the user has no way to tell.
        self._mic_device = QComboBox()
        self._mic_devices: list = []
        try:
            from ai_companion.services.speech_engine import (
                default_input_device,
                list_input_devices,
            )

            self._mic_devices = list_input_devices()
            current_default = default_input_device()
        except Exception:  # noqa: BLE001
            current_default = None

        self._mic_device.addItem("System default", None)
        for index, label in self._mic_devices:
            self._mic_device.addItem(f"[{index}] {label}", index)

        saved = getattr(self._sm.config.speech, "input_device_index", None)
        if saved is not None:
            position = self._mic_device.findData(saved)
            if position >= 0:
                self._mic_device.setCurrentIndex(position)
        speech_form.addRow("Microphone:", self._mic_device)

        if not self._mic_devices:
            warn = QLabel(
                "No microphones detected. Check Windows privacy settings "
                "(Settings > Privacy > Microphone) and that sounddevice is "
                "installed."
            )
            warn.setObjectName("MetaText")
            warn.setWordWrap(True)
            speech_form.addRow("", warn)

        self._mic_test_btn = QPushButton("Test microphone (4s)")
        self._mic_test_btn.setToolTip(
            "Records four seconds, reports the measured level, and "
            "transcribes what it heard."
        )
        self._mic_test_btn.clicked.connect(self._test_microphone)
        speech_form.addRow("", self._mic_test_btn)

        self._mic_result = QLabel("Not tested yet.")
        self._mic_result.setObjectName("MetaText")
        self._mic_result.setWordWrap(True)
        speech_form.addRow("", self._mic_result)

        # ".en" variants are English-only and measurably faster; there is no
        # reason to pay for multilingual weights on an English-only setup.
        self._whisper_model = QComboBox()
        self._whisper_model.addItems([
            "tiny.en", "base.en", "small.en", "tiny", "base", "small", "medium",
        ])
        self._whisper_model.setCurrentText(self._sm.config.speech.whisper_model)
        self._whisper_model.setToolTip(
            "base.en is the recommended balance: ~10x realtime on CPU."
        )
        speech_form.addRow("Listening model:", self._whisper_model)

        # Voice picker, populated from what is actually on disk.
        voice_row = QHBoxLayout()
        self._piper_voice = QComboBox()
        voices: list[str] = []
        voice_service = self._sm.get("Voice")
        if voice_service is not None:
            try:
                voices = voice_service.available_voices()
            except Exception:  # noqa: BLE001
                voices = []
        if voices:
            self._piper_voice.addItems(voices)
            current = getattr(self._sm.config.speech, "piper_voice", "")
            if current in voices:
                self._piper_voice.setCurrentText(current)
        else:
            self._piper_voice.addItem("(none installed)")
            self._piper_voice.setEnabled(False)
        voice_row.addWidget(self._piper_voice, 1)

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setEnabled(bool(voices))
        self._preview_btn.setToolTip("Speak a sample line with this voice.")
        self._preview_btn.clicked.connect(self._preview_voice)
        voice_row.addWidget(self._preview_btn)

        voice_holder = QWidget()
        voice_holder.setLayout(voice_row)
        speech_form.addRow("Voice:", voice_holder)

        if not voices:
            hint = QLabel(
                "No voices found. Run tools/get_speech.py to download them."
            )
            hint.setObjectName("MetaText")
            hint.setWordWrap(True)
            speech_form.addRow("", hint)

        # Engine choice. Speed vs naturalness, and only the user can judge.
        self._tts_engine = QComboBox()
        self._tts_engine.addItem("piper - fast (~10x realtime)", "piper")
        self._tts_engine.addItem(
            "kokoro - slower (~1.2x realtime), more natural", "kokoro"
        )
        current_engine = getattr(
            self._sm.config.speech, "tts_engine_name", "piper"
        )
        position = self._tts_engine.findData(current_engine)
        if position >= 0:
            self._tts_engine.setCurrentIndex(position)
        self._tts_engine.setToolTip(
            "Kokoro sounds more natural but takes about as long to generate "
            "as the reply lasts. Run tools/audition_voices.py to compare."
        )
        speech_form.addRow("Speech engine:", self._tts_engine)

        self._kokoro_voice = QComboBox()
        try:
            from ai_companion.services.speech_engine import KOKORO_VOICES

            for name, description in KOKORO_VOICES:
                self._kokoro_voice.addItem(f"{name} - {description}", name)
        except Exception:  # noqa: BLE001
            self._kokoro_voice.addItem("bm_george", "bm_george")
        current_kokoro = getattr(
            self._sm.config.speech, "kokoro_voice", "bm_george"
        )
        position = self._kokoro_voice.findData(current_kokoro)
        if position >= 0:
            self._kokoro_voice.setCurrentIndex(position)
        speech_form.addRow("Kokoro voice:", self._kokoro_voice)

        # Voice character. This matters more than the voice file itself:
        # Piper's British male is already close in timbre, what was missing
        # was the room - EQ, reverb, slight detune, even level.
        self._voice_fx = QComboBox()
        try:
            from ai_companion.services.voice_fx import describe_presets

            for name, description in describe_presets():
                self._voice_fx.addItem(f"{name} - {description}", name)
        except Exception:  # noqa: BLE001
            self._voice_fx.addItem("jarvis", "jarvis")
        current_fx = getattr(self._sm.config.speech, "voice_fx", "jarvis")
        position = self._voice_fx.findData(current_fx)
        if position >= 0:
            self._voice_fx.setCurrentIndex(position)
        self._voice_fx.setToolTip(
            "Post-processing applied to the voice. Press Preview to hear it."
        )
        speech_form.addRow("Voice character:", self._voice_fx)

        self._wake_word = QLineEdit(
            getattr(self._sm.config.speech, "wake_word", "jarvis")
        )
        self._wake_word.setToolTip(
            "Spoken trigger. Matching is deliberately loose - Whisper often "
            "hears 'Jarvis' as 'Travis'."
        )
        speech_form.addRow("Wake word:", self._wake_word)

        self._require_wake = QCheckBox("Ignore speech without the wake word")
        self._require_wake.setChecked(
            bool(getattr(self._sm.config.speech, "require_wake_word", True))
        )
        speech_form.addRow("", self._require_wake)

        self._follow_up = QDoubleSpinBox()
        self._follow_up.setRange(0.0, 60.0)
        self._follow_up.setSingleStep(1.0)
        self._follow_up.setValue(
            float(getattr(self._sm.config.speech, "follow_up_seconds", 12.0))
        )
        self._follow_up.setSuffix(" s")
        self._follow_up.setToolTip(
            "After it answers, keep listening this long without needing the "
            "wake word again. Set to 0 to require the wake word every time."
        )
        speech_form.addRow("Follow-up window:", self._follow_up)

        self._silence_seconds = QDoubleSpinBox()
        self._silence_seconds.setRange(0.3, 5.0)
        self._silence_seconds.setSingleStep(0.1)
        self._silence_seconds.setValue(
            float(getattr(self._sm.config.speech, "silence_seconds", 1.0))
        )
        self._silence_seconds.setSuffix(" s")
        self._silence_seconds.setToolTip(
            "How long you must stay quiet before the utterance is sent."
        )
        speech_form.addRow("End-of-speech pause:", self._silence_seconds)

        self._silence_threshold = QDoubleSpinBox()
        self._silence_threshold.setRange(0.001, 0.2)
        self._silence_threshold.setSingleStep(0.002)
        self._silence_threshold.setDecimals(3)
        self._silence_threshold.setValue(
            float(getattr(self._sm.config.speech, "silence_threshold", 0.012))
        )
        self._silence_threshold.setToolTip(
            "Mic level below this counts as silence. Raise it in a noisy room."
        )
        speech_form.addRow("Silence threshold:", self._silence_threshold)

        self._tts_rate = QSpinBox()
        self._tts_rate.setRange(50, 300)
        self._tts_rate.setValue(self._sm.config.speech.tts_rate)
        speech_form.addRow("TTS Rate:", self._tts_rate)

        tabs.addTab(speech_tab, "Speech")

        # Code Workspace Settings
        code_tab = QWidget()
        code_form = QFormLayout(code_tab)

        self._exec_timeout = QSpinBox()
        self._exec_timeout.setRange(1, 300)
        self._exec_timeout.setValue(self._sm.config.code_workspace.max_execution_time)
        self._exec_timeout.setSuffix(" sec")
        code_form.addRow("Execution Timeout:", self._exec_timeout)

        self._exec_memory = QSpinBox()
        self._exec_memory.setRange(64, 4096)
        self._exec_memory.setValue(self._sm.config.code_workspace.max_memory_mb)
        self._exec_memory.setSuffix(" MB")
        code_form.addRow("Memory Limit:", self._exec_memory)

        tabs.addTab(code_tab, "Code")

        # Image Gen Settings
        img_tab = QWidget()
        img_form = QFormLayout(img_tab)

        self._comfyui_path = QLineEdit(self._sm.config.image_gen.comfyui_path)
        comfyui_row = QHBoxLayout()
        comfyui_row.addWidget(self._comfyui_path)
        browse_comfy = QPushButton("Browse")
        browse_comfy.clicked.connect(self._browse_comfyui)
        comfyui_row.addWidget(browse_comfy)
        img_form.addRow("ComfyUI Path:", comfyui_row)

        self._img_width = QSpinBox()
        self._img_width.setRange(64, 2048)
        self._img_width.setValue(self._sm.config.image_gen.default_width)
        img_form.addRow("Default Width:", self._img_width)

        self._img_height = QSpinBox()
        self._img_height.setRange(64, 2048)
        self._img_height.setValue(self._sm.config.image_gen.default_height)
        img_form.addRow("Default Height:", self._img_height)

        tabs.addTab(img_tab, "Image Gen")

        layout.addWidget(tabs)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_settings)
        btn_row.addWidget(save_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Files (*.gguf);;All Files (*)"
        )
        if path:
            self._model_path.setText(path)

    def _browse_comfyui(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select ComfyUI Directory")
        if path:
            self._comfyui_path.setText(path)

    def _test_microphone(self) -> None:
        """Record a few seconds and report real numbers.

        Reports the measured level against the threshold rather than a
        pass/fail light: "peak 0.004 vs threshold 0.012" tells the user to
        raise their gain, where "test failed" tells them nothing.
        """
        from ai_companion.services.speech_engine import test_microphone

        self._mic_test_btn.setEnabled(False)
        self._mic_test_btn.setText("Recording... speak now")
        self._mic_result.setText("Listening for 4 seconds - say something.")
        QApplication.processEvents()

        transcriber = None
        voice_service = self._sm.get("Voice")
        if voice_service is not None:
            transcriber = getattr(voice_service, "_transcriber", None)

        try:
            result = test_microphone(
                device_index=self._mic_device.currentData(),
                threshold=float(
                    getattr(self._sm.config.speech, "silence_threshold", 0.012)
                ),
                transcriber=transcriber,
            )
        except Exception as exc:  # noqa: BLE001 - never crash Settings
            result = None
            self._mic_result.setText(f"Test failed: {exc}")
        finally:
            self._mic_test_btn.setText("Test microphone (4s)")
            self._mic_test_btn.setEnabled(True)

        if result is None:
            return
        if not result.ok:
            self._mic_result.setText(result.error or result.verdict)
            return

        message = result.verdict
        if result.transcript:
            message += f'\nHeard: "{result.transcript}"'
        self._mic_result.setText(message)

    def _preview_voice(self) -> None:
        """Speak a sample so the user can choose by ear, not by description."""
        voice_service = self._sm.get("Voice")
        if voice_service is None:
            return
        self._preview_btn.setEnabled(False)
        self._preview_btn.setText("Speaking...")
        QApplication.processEvents()
        try:
            # Apply the selected character first so Preview auditions what is
            # on screen, not what was last saved.
            # Audition exactly what is on screen, not what was last saved.
            self._sm.config.speech.voice_fx = (
                self._voice_fx.currentData() or "natural"
            )
            self._sm.config.speech.tts_engine_name = (
                self._tts_engine.currentData() or "piper"
            )
            self._sm.config.speech.kokoro_voice = (
                self._kokoro_voice.currentData() or "bm_george"
            )
            voice_service._synthesiser = voice_service._build_synthesiser()
            voice_service.preview_voice(self._piper_voice.currentText())
        finally:
            self._preview_btn.setText("Preview")
            self._preview_btn.setEnabled(True)

    def _save_settings(self) -> None:
        """Save settings and reload services as needed."""
        config = self._sm.config

        # Update LLM config.
        #
        # An EMPTY path is never written over a stored one. If the dialog is
        # opened before the field is populated, or the text is cleared by
        # accident, saving would otherwise wipe a working model path and the
        # app would come back reporting "no model loaded" on every launch.
        typed_path = self._model_path.text().strip()
        if typed_path or not config.llm.model_path:
            config.llm.model_path = typed_path
        config.llm.context_length = self._ctx_length.value()
        config.llm.gpu_layers = self._gpu_layers.value()
        config.llm.threads = self._threads.value()
        config.llm.temperature = self._temperature.value()
        config.llm.max_tokens = self._max_tokens.value()
        config.llm.system_prompt = self._system_prompt.toPlainText().strip()

        # Update Vault config. Route a real change through the service so the
        # header badge, the memory-panel warning and config.json all stay in
        # step; skip it entirely when untouched so the badge wins.
        from ai_companion.models.config_models import VaultMode

        chosen = self._vault_mode.currentText()
        if chosen != self._initial_vault_mode:
            vault = self._sm.get("Vault")
            if vault is not None:
                vault.set_mode(VaultMode(chosen), persist=False)
            else:
                config.vault.mode = VaultMode(chosen)
        config.vault.max_file_size_mb = self._max_file_size.value()

        # Update Memory config
        config.memory.auto_extract = self._auto_extract.isChecked()

        # Update Speech config
        config.speech.whisper_model = self._whisper_model.currentText()
        config.speech.tts_rate = self._tts_rate.value()
        if self._piper_voice.isEnabled():
            config.speech.piper_voice = self._piper_voice.currentText()
        config.speech.wake_word = self._wake_word.text().strip()
        config.speech.require_wake_word = self._require_wake.isChecked()
        config.speech.silence_seconds = self._silence_seconds.value()
        config.speech.silence_threshold = self._silence_threshold.value()
        config.speech.input_device_index = self._mic_device.currentData()
        config.speech.follow_up_seconds = self._follow_up.value()
        config.speech.voice_fx = self._voice_fx.currentData() or "natural"
        config.speech.tts_engine_name = (
            self._tts_engine.currentData() or "piper"
        )
        config.speech.kokoro_voice = (
            self._kokoro_voice.currentData() or "bm_george"
        )

        # Update Code config
        config.code_workspace.max_execution_time = self._exec_timeout.value()
        config.code_workspace.max_memory_mb = self._exec_memory.value()

        # Update Image Gen config
        config.image_gen.comfyui_path = self._comfyui_path.text()
        config.image_gen.default_width = self._img_width.value()
        config.image_gen.default_height = self._img_height.value()

        # Save to disk
        from ai_companion.config import ConfigManager
        cm = ConfigManager()
        cm._config = config
        cm.save()

        # Reload if the path changed OR nothing is currently loaded.
        # The second half matters: if the startup load failed, the path in the
        # dialog already equals the saved path, so a "changed?" test alone
        # would make Save a no-op and leave the user with no way to retry.
        new_path = config.llm.model_path.strip()
        llm = self._sm.get("LLM")
        already_loaded = bool(getattr(llm, "model_loaded", False))
        if new_path and (new_path != self._initial_model_path or not already_loaded):
            if llm is not None:
                try:
                    llm.load_model(new_path)
                except Exception as exc:  # noqa: BLE001 - surfaced to user
                    QMessageBox.warning(
                        self, "Model load failed", f"Could not load model:\n{exc}"
                    )

        self.accept()
