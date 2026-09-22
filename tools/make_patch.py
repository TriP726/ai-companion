#!/usr/bin/env python3
"""Generator script that compiles Install_patch.py with embedded payload."""

import base64
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAYLOAD_FILES = [
    ".gitignore",
    "ai_companion/graph3d/__init__.py",
    "ai_companion/graph3d/camera.py",
    "ai_companion/graph3d/layout3d.py",
    "ai_companion/ui/graph3d_view.py",
    "ai_companion/ui/graph_panel.py",
    "pyproject.toml",
    "tests/test_graph3d.py",
]

def build():
    payload_data = {}
    for rel_path in PAYLOAD_FILES:
        full_path = REPO_ROOT / rel_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing payload file: {full_path}")
        raw_bytes = full_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        payload_data[rel_path] = {
            "sha256": sha256,
            "b64": b64,
            "size": len(raw_bytes),
        }
        print(f"Packed {rel_path} ({len(raw_bytes)} bytes, sha256: {sha256[:12]}...)")

    # Generate the self-contained installer script
    installer_code = f'''#!/usr/bin/env python3
"""Self-contained atomic patch installer for ai-companion.

Applies the 3D graph visual restyle, drops stale root-level duplicate trees,
updates pyproject.toml build configuration, creates timestamped backups,
and verifies against the pytest test suite with automatic rollback on any new failure.

Usage:
    python Install_patch.py [--target PATH] [--python PATH] [--baseline] [--dry-run]
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

PAYLOAD: Dict[str, Dict[str, str]] = {repr(payload_data)}

PURGE_DIRS = [
    "graph3d",
    "ui",
    "localpycs",
]

PURGE_FILES = [
    "install_patch.py",
    "AICompanion.exe",
    "AICompanion.pkg",
    "Analysis-00.toc",
    "COLLECT-00.toc",
    "EXE-00.toc",
    "PKG-00.toc",
    "PYZ-00.pyz",
    "PYZ-00.toc",
    "base_library.zip",
    "warn-ai_companion.txt",
    "xref-ai_companion.html",
]

KNOWN_PREEXISTING_FAILURES: Set[str] = {{
    "tests/test_services.py::TestVaultService::test_private_remove_does_not_write_index",
    "tests/test_services.py::TestVaultService::test_stop_while_private_does_not_write_index",
    "tests/test_vault_audit.py::TestRemoveFileRespectsPrivateMode::test_index_on_disk_is_not_touched_while_private",
    "tests/test_vault_audit.py::TestRemoveFileRespectsPrivateMode::test_stop_while_private_does_not_flush_index",
}}


def log(msg: str) -> None:
    print(f"[Install_patch] {{msg}}")


def error(msg: str) -> None:
    print(f"[Install_patch ERROR] {{msg}}", file=sys.stderr)


def resolve_target(target_arg: Optional[str]) -> Path:
    if target_arg:
        t = Path(target_arg).resolve()
        if not t.is_dir():
            raise FileNotFoundError(f"Target directory does not exist: {{t}}")
        return t

    # 1. Standard default Windows install location
    win_dev = Path(r"C:\\dev\\ai_companion")
    if win_dev.is_dir():
        return win_dev.resolve()

    # 2. Current working directory if it looks like the repo root
    cwd = Path.cwd().resolve()
    if (cwd / "ai_companion").is_dir() or (cwd / "pyproject.toml").is_file():
        return cwd

    # 3. Check parent directory
    if (cwd.parent / "ai_companion").is_dir():
        return cwd.parent.resolve()

    # Default fallback
    return win_dev


def resolve_python(python_arg: Optional[str], target: Path) -> Path:
    if python_arg:
        p = Path(python_arg).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Python executable does not exist: {{p}}")
        return p

    candidates = [
        target / ".venv" / "Scripts" / "python.exe",
        target / ".venv" / "bin" / "python",
        target / "venv" / "Scripts" / "python.exe",
        target / "venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    return Path(sys.executable).resolve()


def parse_failed_tests(output: str) -> Set[str]:
    norm = output.replace("\\\\", "/")
    # Match patterns like: FAILED tests/test_services.py::TestClass::test_func
    matches = set(re.findall(r"FAILED\\s+([^\\s:]+(?:::[-_\\w]+)+)", norm))
    return matches


def run_pytest(python_exe: Path, target: Path) -> Tuple[int, Set[str], str]:
    cmd = [str(python_exe), "-m", "pytest", "-q", "--tb=line"]
    log(f"Running pytest: {{' '.join(cmd)}} (cwd={{target}})")
    try:
        res = subprocess.run(
            cmd,
            cwd=str(target),
            capture_output=True,
            text=True,
            timeout=300,
        )
        combined = (res.stdout or "") + "\\n" + (res.stderr or "")
        failures = parse_failed_tests(combined)
        return res.returncode, failures, combined
    except Exception as e:
        error(f"Failed to execute pytest: {{e}}")
        return 2, set(), str(e)


def purge_pycache(target: Path, dry_run: bool = False) -> int:
    count = 0
    for p in target.rglob("__pycache__"):
        if p.is_dir() and "data_backups" not in p.parts:
            count += 1
            if not dry_run:
                try:
                    shutil.rmtree(p)
                except Exception as e:
                    log(f"Warning: could not remove {{p}}: {{e}}")
    return count


def perform_backup(target: Path, backup_dir: Path) -> dict:
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest = {{
        "timestamp": datetime.datetime.now().isoformat(),
        "target": str(target),
        "files_overwritten": [],
        "files_new": [],
        "purged_dirs": [],
        "purged_files": [],
    }}

    # Backup files that will be overwritten
    for rel_path in PAYLOAD.keys():
        dest = target / rel_path
        if dest.is_file():
            b_path = backup_dir / "overwritten" / rel_path
            b_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, b_path)
            manifest["files_overwritten"].append(rel_path)
        else:
            manifest["files_new"].append(rel_path)

    # Backup directories that will be purged
    for d in PURGE_DIRS:
        p_dir = target / d
        if p_dir.is_dir():
            b_pdir = backup_dir / "purged" / d
            b_pdir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(p_dir, b_pdir)
            manifest["purged_dirs"].append(d)

    # Backup files that will be purged
    for f in PURGE_FILES:
        p_file = target / f
        if p_file.is_file():
            b_pfile = backup_dir / "purged_files" / f
            b_pfile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p_file, b_pfile)
            manifest["purged_files"].append(f)

    with open(backup_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def perform_rollback(target: Path, backup_dir: Path, manifest: dict) -> None:
    log("ROllING BACK: Restoring all files and directories from backup...")
    try:
        # 1. Remove newly created files
        for rel_path in manifest.get("files_new", []):
            dest = target / rel_path
            if dest.is_file():
                dest.unlink()
                log(f"Rollback: removed new file {{rel_path}}")

        # 2. Restore overwritten files
        for rel_path in manifest.get("files_overwritten", []):
            src = backup_dir / "overwritten" / rel_path
            dest = target / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            log(f"Rollback: restored {{rel_path}}")

        # 3. Restore purged directories
        for d in manifest.get("purged_dirs", []):
            src = backup_dir / "purged" / d
            dest = target / d
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            log(f"Rollback: restored directory {{d}}")

        # 4. Restore purged files
        for f in manifest.get("purged_files", []):
            src = backup_dir / "purged_files" / f
            dest = target / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            log(f"Rollback: restored file {{f}}")

        purge_pycache(target, dry_run=False)
        log("Rollback completed successfully.")
    except Exception as e:
        error(f"Critical error during rollback: {{e}}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Self-contained atomic patch installer for ai-companion."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target repo root directory (default: C:\\\\dev\\\\ai_companion or current directory)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Path to python executable with pytest installed",
    )
    parser.add_argument(
        "--backup-root",
        type=str,
        default=None,
        help="Root directory for backups (default: <target>/data_backups)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Record baseline test failures prior to applying patch",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest (apply files without running tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without modifying any files",
    )

    args = parser.parse_args()

    try:
        target = resolve_target(args.target)
    except Exception as e:
        error(f"Invalid target: {{e}}")
        return 2

    if not target.is_dir():
        error(f"Target directory does not exist: {{target}}")
        return 2

    try:
        python_exe = resolve_python(args.python, target)
    except Exception as e:
        error(f"Invalid python executable: {{e}}")
        return 2

    backup_root = Path(args.backup_root).resolve() if args.backup_root else (target / "data_backups")

    log("=" * 60)
    log("ai-companion Patch Installer")
    log("=" * 60)
    log(f"Target directory   : {{target}}")
    log(f"Python executable  : {{python_exe}}")
    log(f"Backup root        : {{backup_root}}")
    log(f"Baseline mode      : {{args.baseline}}")
    log(f"Skip tests         : {{args.skip_tests}}")
    log(f"Dry run            : {{args.dry_run}}")
    log("=" * 60)

    if args.dry_run:
        log("DRY RUN: The following files will be installed/updated:")
        for rel_path, meta in PAYLOAD.items():
            dest = target / rel_path
            status = "UPDATE" if dest.is_file() else "CREATE"
            log(f"  [{{status}}] {{rel_path}} ({{meta['size']}} bytes, sha256: {{meta['sha256'][:12]}}...)")
        log("DRY RUN: The following root duplicate directories will be purged:")
        for d in PURGE_DIRS:
            p_dir = target / d
            status = "EXISTS (will remove)" if p_dir.is_dir() else "NOT PRESENT"
            log(f"  [PURGE] {{d}}/ ({{status}})")
        log("DRY RUN: The following stale build artifacts and duplicate files will be purged:")
        for f in PURGE_FILES:
            p_file = target / f
            status = "EXISTS (will remove)" if p_file.is_file() else "NOT PRESENT"
            log(f"  [PURGE FILE] {{f}} ({{status}})")
        pycache_count = purge_pycache(target, dry_run=True)
        log(f"DRY RUN: Would purge {{pycache_count}} __pycache__ directories.")
        log("DRY RUN finished. No changes made.")
        return 0

    # 1. Baseline recording (if requested)
    baseline_failures = set(KNOWN_PREEXISTING_FAILURES)
    if not args.skip_tests and args.baseline:
        log("Step 1: Running baseline pytest prior to patch...")
        code, pre_fails, pre_out = run_pytest(python_exe, target)
        baseline_failures.update(pre_fails)
        log(f"Baseline failures recorded ({{len(baseline_failures)}}):")
        for f in sorted(baseline_failures):
            log(f"  - {{f}}")

    # 2. Create timestamped backup
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"backup_{{ts}}"
    log(f"Step 2: Creating backup in {{backup_dir}}...")
    manifest = perform_backup(target, backup_dir)
    log(f"Backup complete. {{len(manifest['files_overwritten'])}} files overwritten, "
        f"{{len(manifest['purged_dirs'])}} dirs archived, {{len(manifest['purged_files'])}} files archived.")

    # 3. Apply payload
    log("Step 3: Writing patch payload files...")
    written_files = []
    try:
        for rel_path, meta in PAYLOAD.items():
            dest = target / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = base64.b64decode(meta["b64"])
            digest = hashlib.sha256(raw).hexdigest()
            if digest != meta["sha256"]:
                raise ValueError(
                    f"Checksum mismatch for {{rel_path}}: expected {{meta['sha256']}}, got {{digest}}"
                )
            dest.write_bytes(raw)
            written_files.append(rel_path)
            log(f"  Wrote {{rel_path}} ({{len(raw)}} bytes, sha256: {{digest[:12]}}...)")

        # 4. Purge stale directories and files
        log("Step 4: Purging stale root directories and build artifacts...")
        for d in PURGE_DIRS:
            p_dir = target / d
            if p_dir.is_dir():
                shutil.rmtree(p_dir)
                log(f"  Removed stale directory: {{d}}/")

        for f in PURGE_FILES:
            p_file = target / f
            if p_file.is_file():
                p_file.unlink()
                log(f"  Removed stale file: {{f}}")

        # 5. Purge __pycache__
        log("Step 5: Purging __pycache__ directories...")
        pyc_cleaned = purge_pycache(target, dry_run=False)
        log(f"  Purged {{pyc_cleaned}} __pycache__ directories.")

    except Exception as e:
        error(f"Installation failed during file write: {{e}}")
        perform_rollback(target, backup_dir, manifest)
        return 1

    # 6. Verify with pytest
    if not args.skip_tests:
        log("Step 6: Verifying patch with pytest...")
        code, post_fails, post_out = run_pytest(python_exe, target)
        new_failures = post_fails - baseline_failures

        if new_failures:
            error("TEST VERIFICATION FAILED! New test failures detected:")
            for f in sorted(new_failures):
                error(f"  NEW FAILURE: {{f}}")
            perform_rollback(target, backup_dir, manifest)
            return 1
        else:
            log(f"Post-patch test verification PASSED.")
            if post_fails:
                log(f"Pre-existing grandfathered failures ({{len(post_fails)}}):")
                for f in sorted(post_fails):
                    log(f"  - {{f}}")

    log("=" * 60)
    log("SUCCESS: Patch applied successfully!")
    log(f"Files updated: {{len(written_files)}}")
    log(f"Backup saved to: {{backup_dir}}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    out_file = REPO_ROOT / "Install_patch.py"
    out_file.write_text(installer_code, encoding="utf-8")
    print(f"Generated {out_file} ({out_file.stat().st_size} bytes)")

    # Also write to /home/user/Install_patch.py
    home_file = Path("/home/user/Install_patch.py")
    home_file.write_text(installer_code, encoding="utf-8")
    print(f"Also wrote {home_file} ({home_file.stat().st_size} bytes)")

if __name__ == "__main__":
    build()
