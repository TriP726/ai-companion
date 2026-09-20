"""AI Companion - self-contained patch installer.

Single file. No zip handling required.

Fixes two security gaps found while reviewing vault_service.py and
image_gen_service.py against the app's own stated security guarantees:

1. vault_service.py
   - remove_file() wrote the on-disk vault index unconditionally, even
     while the Vault was in PRIVATE mode. Now deferred while Private and
     flushed automatically on return to Normal mode. The physical file
     delete itself still always happens immediately either way.
   - get_file_path() built its path with plain concatenation and an
     .exists() check only - no containment check at all, unlike every
     other method in that file. A crafted vault_path containing "../"
     segments could resolve to a file anywhere on disk. Nothing in the
     shipped UI calls this yet, but it's a public method. Fixed to
     validate through PathValidator like the rest of the file.

2. image_gen_service.py
   - start_worker() passed config.image_gen.host straight through to
     the ComfyUI worker's --listen flag, and _submit_to_worker() used
     the same config value to build the request URL. The class
     docstring claimed "binds to 127.0.0.1 only", but nothing enforced
     that - it was only true because the config default happened to be
     127.0.0.1. Fixed: both bind and connect are now pinned to a
     hardcoded loopback constant, so a bad config value can no longer
     put the worker on the network. A status message is emitted if the
     configured value is ever different, so it's visible rather than
     silently ignored.

Also adds tests/test_vault_audit.py (10 tests) and
tests/test_image_gen_audit.py (4 tests) pinning both fixes.

USAGE
-----
Save this file into  C:\\dev\\ai_companion\\  then run:

    cd C:\\dev\\ai_companion
    .venv\\Scripts\\python.exe install_patch.py

It backs up your current files, writes the patched ones, and runs the
full test suite. Expected result: 452 passed (438 existing + 14 new).
"""
from __future__ import annotations

import base64
import io
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PAYLOAD = """\
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
\
"""

# Maps: path inside the payload zip -> path in the project, relative to
# this script's own location (which should be the project root).
FILES: dict[str, str] = {
    "vault_service.py": "ai_companion/services/vault_service.py",
    "image_gen_service.py": "ai_companion/services/image_gen_service.py",
    "test_vault_audit.py": "tests/test_vault_audit.py",
    "test_image_gen_audit.py": "tests/test_image_gen_audit.py",
}


def main() -> int:
    root = Path(__file__).resolve().parent
    print("AI Companion patch installer")
    print("=" * 64)
    print(f"Project root: {root}")
    print()

    try:
        raw = base64.b64decode(PAYLOAD)
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception as exc:
        print(f"Could not read embedded payload: {exc}")
        return 1

    backup_dir = root / "patches_backup" / datetime.now().strftime("%Y%m%d-%H%M%S")

    written: list[Path] = []
    for payload_name, target_rel in FILES.items():
        target = root / target_rel
        try:
            data = zf.read(payload_name)
        except KeyError:
            print(f"  MISSING from payload: {payload_name} - skipping")
            continue

        if target.exists():
            backup_path = backup_dir / target_rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_path)
            print(f"  backed up  {target_rel}")
        else:
            print(f"  new file   {target_rel}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(target)
        print(f"  wrote      {target_rel}")

    if not written:
        print()
        print("Nothing was written. Aborting before running tests.")
        return 1

    print()
    if backup_dir.exists():
        print(f"Previous versions backed up to: {backup_dir}")
    print()
    print("Running the test suite...")
    print("-" * 64)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/"],
        cwd=str(root),
    )

    print("-" * 64)
    if result.returncode == 0:
        print("All tests passed.")
        print("Expected: 452 passed (438 existing + 14 new).")
        return 0

    print("Tests did not pass cleanly.")
    if backup_dir.exists():
        print(f"Your previous code is intact in {backup_dir}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
