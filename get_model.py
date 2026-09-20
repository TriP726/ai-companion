"""Download a better GGUF model and point the app at it.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe get_model.py

Downloads over HTTPS from huggingface.co with a resumable request, verifies
the file size, and offers to update config.json. Nothing is downloaded
automatically - you pick from a menu.
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co"

MODELS = [
    {
        "key": "1",
        "name": "Qwen2.5-7B-Instruct (Q4_K_M)",
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "gb": 4.36,
        "ram": "~6.5 GB while running",
        "why": "Best all-round upgrade. Much stronger reasoning and "
               "instruction-following than Phi-3-mini. 32K context.",
    },
    {
        "key": "2",
        "name": "Qwen2.5-Coder-7B-Instruct (Q4_K_M)",
        "repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "gb": 4.36,
        "ram": "~6.5 GB while running",
        "why": "Same size, tuned for code. Best pick if most of your chats "
               "are programming questions like the Discord bot work.",
    },
    {
        "key": "3",
        "name": "Qwen2.5-7B-Instruct (Q5_K_M, higher quality)",
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
        "gb": 5.07,
        "ram": "~7.5 GB while running",
        "why": "Slightly better quality than Q4_K_M, slightly slower. "
               "Only if you have RAM headroom.",
    },
    {
        "key": "4",
        "name": "Llama-3.1-8B-Instruct (Q4_K_M)",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "gb": 4.58,
        "ram": "~6.8 GB while running",
        "why": "Alternative to Qwen. More natural English prose, slightly "
               "weaker at code. Pick this if Qwen's style annoys you.",
    },
]


def human(n: int) -> str:
    return f"{n / 1024 / 1024 / 1024:.2f} GB"


def download(url: str, dest: Path) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "ai-companion"})
    if existing:
        print(f"   resuming from {human(existing)}")
        req.add_header("Range", f"bytes={existing}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + existing
            mode = "ab" if existing else "wb"
            done = existing
            with open(tmp, mode) as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        bar = "#" * int(pct / 2.5)
                        print(
                            f"\r   [{bar:<40}] {pct:5.1f}%  "
                            f"{human(done)} / {human(total)}",
                            end="",
                            flush=True,
                        )
            print()
    except KeyboardInterrupt:
        print("\n   Interrupted. Re-run to resume where it stopped.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"\n   Download failed: {exc}")
        print("   Re-run to resume, or download manually from the URL above.")
        return False

    tmp.replace(dest)
    return True


def main() -> int:
    root = Path(__file__).resolve().parent
    print("=" * 66)
    print("  DOWNLOAD A BETTER MODEL")
    print("=" * 66)
    print("\nYour current model is Phi-3-mini (3.8B, 2.2 GB).")
    print("All options below are bigger and noticeably more capable.\n")

    for m in MODELS:
        print(f"  [{m['key']}] {m['name']}")
        print(f"      download {m['gb']} GB   {m['ram']}")
        print(f"      {m['why']}\n")
    print("  [q] quit\n")

    choice = input("Which model? ").strip().lower()
    if choice == "q":
        return 0
    model = next((m for m in MODELS if m["key"] == choice), None)
    if model is None:
        print("Unknown option.")
        return 1

    target_dir = Path("C:/models") if sys.platform == "win32" else root / "models"
    raw = input(f"\nSave folder [{target_dir}]: ").strip()
    if raw:
        target_dir = Path(raw)
    target_dir.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(target_dir).free
    need = int(model["gb"] * 1024**3 * 1.05)
    print(f"\nFree space: {human(free)}   needed: ~{human(need)}")
    if free < need:
        print("Not enough free disk space.")
        return 1

    dest = target_dir / model["file"]
    if dest.exists():
        print(f"\nAlready downloaded: {dest}")
    else:
        url = f"{BASE}/{model['repo']}/resolve/main/{model['file']}?download=true"
        print(f"\nDownloading from:\n  {url}\n")
        if not download(url, dest):
            return 1

    size = dest.stat().st_size
    print(f"\nFile: {dest}")
    print(f"Size: {human(size)}")
    expected = model["gb"] * 1024**3
    if abs(size - expected) / expected > 0.05:
        print("WARNING: size differs from expected - the file may be truncated.")
        print("Delete it and re-run if the model fails to load.")

    cfg_path = root / "config.json"
    if cfg_path.exists():
        ans = input("\nPoint config.json at this model? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg.setdefault("llm", {})
            cfg["llm"]["model_path"] = str(dest).replace("\\", "/")
            # 7B at 4096 context is comfortable in 16 GB; keep threads as set.
            cfg["llm"]["context_length"] = 4096
            cfg_path.write_text(
                json.dumps(cfg, indent=2), encoding="utf-8"
            )
            print("   config.json updated (context_length set to 4096).")
            print("\nStart the app with run.bat - it will load on startup.")
    else:
        print("\nNo config.json here. Set the path in Settings -> LLM.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
