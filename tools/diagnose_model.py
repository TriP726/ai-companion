"""Standalone model-load diagnostic.

Run from the project root with the venv active:
    python diagnose_model.py

Checks the config path, the file on disk, and llama-cpp-python, then attempts
a real load and prints the full traceback if it fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    print("=" * 62)
    print("AI COMPANION - MODEL LOAD DIAGNOSTIC")
    print("=" * 62)

    print(f"\n[1] Python: {sys.version.split()[0]}")
    print(f"    Executable: {sys.executable}")
    if ".venv" not in sys.executable:
        print("    WARNING: not running inside .venv")

    print("\n[2] llama-cpp-python")
    try:
        import llama_cpp

        print(f"    OK - version {llama_cpp.__version__}")
    except ImportError as exc:
        print(f"    MISSING - {exc}")
        print("    Fix: pip install llama-cpp-python --extra-index-url \\")
        print("         https://abetlen.github.io/llama-cpp-python/whl/cpu")
        return 1

    print("\n[3] config.json")
    cfg_path = Path("config.json")
    if not cfg_path.exists():
        print("    Not found in current directory.")
        print(f"    cwd = {Path.cwd()}")
        return 1
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    llm_cfg = cfg.get("llm", {})
    model_path = llm_cfg.get("model_path", "")
    print(f"    model_path     = {model_path!r}")
    print(f"    context_length = {llm_cfg.get('context_length')}")
    print(f"    threads        = {llm_cfg.get('threads')}")
    print(f"    gpu_layers     = {llm_cfg.get('gpu_layers')}")

    if not model_path:
        print("    EMPTY - set it in Settings, or the app cannot load anything.")
        return 1

    print("\n[4] Model file")
    p = Path(model_path)
    print(f"    exists   = {p.exists()}")
    if not p.exists():
        parent = p.parent
        print(f"    parent   = {parent} (exists: {parent.exists()})")
        if parent.exists():
            found = list(parent.glob("*.gguf"))
            print(f"    .gguf files actually in that folder: "
                  f"{[f.name for f in found] or 'none'}")
        return 1
    print(f"    is_file  = {p.is_file()}")
    print(f"    suffix   = {p.suffix}")
    size_gb = p.stat().st_size / (1024 ** 3)
    print(f"    size     = {size_gb:.2f} GB")
    if size_gb < 0.1:
        print("    WARNING: suspiciously small - truncated download?")

    print("\n[5] Attempting real load (this takes a few seconds)...")
    try:
        from llama_cpp import Llama

        model = Llama(
            model_path=str(p),
            n_ctx=int(llm_cfg.get("context_length", 2048)),
            n_threads=int(llm_cfg.get("threads", 6)),
            n_gpu_layers=int(llm_cfg.get("gpu_layers", 0)),
            verbose=False,
        )
        out = model.create_chat_completion(
            messages=[{"role": "user", "content": "Say OK."}],
            max_tokens=10,
        )
        print("    SUCCESS")
        print(f"    reply = {out['choices'][0]['message']['content']!r}")
        return 0
    except Exception:
        import traceback

        print("    FAILED - full traceback:\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
