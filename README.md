# AI Companion — Local-First Desktop AI

A local-first Windows desktop AI companion built with Python and PySide6. No cloud services, no telemetry — everything runs on your machine.

## Features

### Core AI
- **Local LLM Chat** — Streaming conversation via llama-cpp-python with configurable GGUF models
- **Conversation History** — Multiple conversations with full message history
- **Attachments** — File attachments to messages
- **Cancellation** — Stop generation mid-stream
- **Model Status** — Visible loading/running/error state

### Memory & Knowledge
- **Private Vault** — Secure local file memory; in private mode disables persistent memories, file copies, and audit writes
- **Long-Term Memories** — Owner-reviewed with editing, pinning, deletion, confidence scores, and provenance tracking
- **Knowledge Graph** — Visual searchable graph connecting memories, chats, files, people, projects, and ideas
- **Force-Directed Layout** — Automatic graph positioning with interactive node movement

### Media & Input
- **Camera Access** — Manual capture only; no automatic recording; camera opens on-demand and closes after capture
- **Image Generation** — Local ComfyUI/SDXL worker management with placeholder generation
- **Speech I/O** — Local Whisper STT and system TTS (no cloud APIs)

### Workspaces
- **File Workspace** — Permission-controlled with approved-folder boundaries and audited operations
- **Code Workspace** — Sandboxed Python execution with snapshots, diffs, copy-in/copy-out, and network isolation
- **3D Sandbox** — GLB, GLTF, and OBJ import with scene management and transform editing

## Security Model

- **No cloud services or telemetry** — Everything is local
- **Loopback-only servers** — LLM and ComfyUI bind to 127.0.0.1
- **Path validation** — All filesystem access goes through centralized PathValidator
- **Approved-folder boundaries** — No arbitrary host-file access
- **Sandboxed workers** — Code runs in subprocess with no network, time/memory limits
- **Input validation** — All paths, file types, sizes, and numerical inputs are validated
- **Explicit destructive operations** — No silent data loss

## Architecture

```
Service-Oriented Architecture (Qt Signals/Slots)
├── infrastructure/
│   ├── ServiceManager    — Lifecycle management for all services
│   ├── SignalBus         — Typed Qt signal bus for inter-service communication
│   ├── PathValidator     — Centralized path security
│   ├── JsonStore         — Thread-safe JSON persistence with atomic writes
│   └── BaseService       — Abstract base with start/stop/error lifecycle
├── services/
│   ├── LlmService        — llama.cpp inference with streaming
│   ├── VaultService      — Private file memory
│   ├── MemoryService     — Owner-reviewed long-term memories
│   ├── GraphService      — Knowledge graph with force-directed layout
│   ├── CameraService     — Manual frame capture (no recording)
│   ├── ImageGenService   — ComfyUI worker management
│   ├── SpeechService     — Whisper STT + system TTS
│   ├── FileWorkspace     — Approved-folder file operations
│   ├── CodeWorkspace     — Sandboxed code execution
│   └── Sandbox3DService  — 3D scene management
└── ui/
    ├── MainWindow        — Tab navigation, menu bar, status bar
    ├── ChatPanel         — Streaming chat with attachments
    ├── MemoryPanel       — Memory CRUD with approval workflow
    ├── GraphPanel        — Interactive knowledge graph visualization
    ├── WorkspacePanel    — File browser + code editor
    └── SettingsPanel     — Configuration dialog
```

## Quick Start

### Prerequisites
- Python 3.10+
- Windows 10/11 (primary target)

### Install
```bash
pip install -r requirements.txt
# Or: pip install -e ".[dev]"
```

### Run
```bash
python -m ai_companion.main
```

### Configure a Model
1. Download a GGUF model (e.g., from HuggingFace)
2. Open Settings (Ctrl+,)
3. Set the model path under LLM tab
4. Click Save — the model loads automatically

### Run Tests
```bash
python -m pytest tests/ -v
```

### Build Executable
```bash
pip install pyinstaller
pyinstaller ai_companion.spec
# Output: dist/AICompanion.exe
```

## Project Structure

```
├── ai_companion/
│   ├── main.py              # Entry point
│   ├── config.py            # Configuration management
│   ├── infrastructure/      # Core framework
│   ├── services/            # Business logic services
│   ├── models/              # Data models (Pydantic)
│   └── ui/                  # PySide6 GUI
├── tests/                   # pytest test suite
├── assets/                  # Application icons
├── config.json              # Runtime configuration
├── ai_companion.spec        # PyInstaller packaging spec
├── architecture.md          # Detailed architecture document
└── pyproject.toml           # Project metadata
```

## Configuration

All settings are in `config.json` and can be modified via the Settings dialog (Ctrl+,).

Key settings:
- `llm.model_path` — Path to your GGUF model file
- `llm.context_length` — Context window size (512-131072)
- `llm.gpu_layers` — Number of layers to offload to GPU
- `vault.mode` — "private" (restricted) or "normal"
- `code_workspace.allow_network` — Must be False (security)

## Milestones Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1: Core Shell & Infrastructure | ✅ Done | Service manager, config, JSON store, path validation, basic UI shell |
| M2: LLM Chat Service | ✅ Done | Streaming, cancellation, conversation history, model status |
| M3: Private Vault & Memory | ✅ Done | Vault modes, memory CRUD, approval workflow, pinning |
| M4: Knowledge Graph | ✅ Done | Node/edge CRUD, search, force-directed layout, visualization |
| M5: Camera & Image Gen | ✅ Done | Manual capture, ComfyUI worker, placeholder generation |
| M6: Speech I/O | ✅ Done | Whisper STT, system TTS, audio controls |
| M7: File & Code Workspaces | ✅ Done | Approved folders, sandboxed execution, snapshots, diffs |
| M8: 3D Sandbox | ✅ Done | GLB/GLTF/OBJ import, scene management, transforms |
| M9: Packaging | ✅ Done | PyInstaller spec, app icon, dark theme |

## License

MIT
