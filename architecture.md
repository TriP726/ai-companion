# AI Companion — Architecture & Milestone Plan

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PySide6 GUI Shell                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ ChatPanel│ │MemPanel  │ │GraphPanel│ │ SettingsPanel  │  │
│  │          │ │          │ │          │ │               │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬───────┘  │
│       │             │            │               │          │
│  ┌────┴─────────────┴────────────┴───────────────┴───────┐  │
│  │              ServiceManager (Qt Signal Bus)            │  │
│  └──┬─────┬──────┬──────┬──────┬──────┬──────┬──────┬───┘  │
│     │     │      │      │      │      │      │      │      │
│  ┌──┴──┐┌─┴───┐┌─┴──┐┌──┴──┐┌──┴──┐┌──┴──┐┌──┴──┐┌──┴──┐ │
│  │ LLM ││Vault││Mem ││Graph││Cam  ││ImgGen││Speech││Code │ │
│  │Svc  ││Svc  ││Svc ││Svc  ││Svc  ││ Svc  ││ Svc ││WS   │ │
│  └─────┘└─────┘└────┘└─────┘└─────┘└──────┘└─────┘└─────┘ │
│     │     │      │      │      │      │      │      │      │
│  ┌──┴──┐┌─┴───┐┌─┴──┐┌──┴──┐┌──┴──┐┌──┴──┐┌──┴──┐┌──┴──┐ │
│  │llama││Local││JSON│ │Netw ││OpenCV││Comfy││Whisp││Proc │ │
│  │.cpp ││Files││Store│ │ork  ││     ││UI   ││er   ││Sbox │ │
│  └─────┘└─────┘└────┘└─────┘└─────┘└─────┘└─────┘└─────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Service-oriented**: Each subsystem is a `QObject` with Qt signals/slots, registered in a central `ServiceManager`
2. **Typed Python**: Full type hints, dataclasses/Pydantic models for data, enums for states
3. **Loopback-only servers**: All local servers (LLM, ComfyUI) bind to 127.0.0.1
4. **Path validation**: Centralized `PathValidator` enforces approved-folder boundaries
5. **JSON persistence**: Each service owns its own JSON store with schema versioning
6. **Worker isolation**: Code execution runs in subprocess with restricted filesystem and no network
7. **Signal-based UI updates**: No polling; services emit signals that UI panels connect to

## Milestones

### M1: Core Shell & Infrastructure ✅ IN PROGRESS
- [x] Project structure, pyproject.toml, requirements
- [x] ServiceManager, BaseService, SignalBus
- [x] PathValidator, config system, JSON persistence layer
- [x] Basic PySide6 shell with tab navigation
- [x] Tests for infrastructure

### M2: LLM Chat Service
- [ ] LlamaCppService wrapping llama-cpp-python
- [] Streaming with cancellation
- [ ] Conversation history management
- [ ] Model status tracking
- [ ] ChatPanel UI with streaming display

### M3: Private Vault & Memory
- [ ] VaultService: file memory, no persistence mode
- [ ] MemoryService: owner-reviewed memories with CRUD
- [ ] Confidence scores, provenance, pinning
- [ ] MemoryPanel UI

### M4: Knowledge Graph
- [ ] GraphService: entity/relationship store
- [ ] Visual graph rendering (QGraphicsView)
- [ ] Search and filtering

### M5: Camera & Image Generation
- [ ] CameraService: manual capture, no recording
- [ ] ImageGenService: ComfyUI worker management
- [ ] UI panels for both

### M6: Speech I/O
- [ ] SpeechService: STT (Whisper) and TTS
- [ ] Audio controls in chat

### M7: File & Code Workspaces
- [ ] FileWorkspace: approved folders, audited ops
- [ ] CodeWorkspace: subprocess sandbox, snapshots, diffs

### M8: 3D Sandbox
- [ ] GLB/GLTF/OBJ import
- [ ] Basic 3D viewport (OpenGL or similar)

### M9: Security Hardening & Packaging
- [ ] Input validation everywhere
- [ ] PyInstaller packaging with icon
- [ ] Full test suite pass

## Data Flow Example: Chat Message

```
User types message → ChatPanel.on_send()
  → LlmService.send_message(conversation_id, text, attachments)
    → validates inputs
    → calls llama.cpp with streaming
    → emits chunk_received(str) per token
    → emits generation_complete() when done
  → ChatPanel receives chunks, appends to display
  → MemoryService.extract_candidate(text) if enabled
```

## File Structure

```
ai_companion/
├── main.py
├── config.py
├── infrastructure/
│   ├── service_manager.py
│   ├── base_service.py
│   ├── path_validator.py
│   ├── json_store.py
│   └── signal_bus.py
├── services/
│   ├── llm_service.py
│   ├── vault_service.py
│   ├── memory_service.py
│   ├── graph_service.py
│   ├── camera_service.py
│   ├── image_gen_service.py
│   ├── speech_service.py
│   ├── file_workspace.py
│   ├── code_workspace.py
│   └── sandbox3d_service.py
├── ui/
│   ├── main_window.py
│   ├── chat_panel.py
│   ├── memory_panel.py
│   ├── graph_panel.py
│   ├── settings_panel.py
│   └── widgets/
│       ├── streaming_text.py
│       ├── model_status.py
│       └── attachment_area.py
├── models/
│   ├── conversation.py
│   ├── memory.py
│   ├── graph.py
│   └── config_models.py
└── tests/
    ├── test_infrastructure.py
    ├── test_llm_service.py
    ├── test_vault.py
    ├── test_memory.py
    └── ...
```
