# One failing test — deleted, because it should never have existed

    cd C:\dev\ai_companion
    .venv\Scripts\python.exe install_patch.py
    .venv\Scripts\python.exe build_exe.py
    .venv\Scripts\python.exe verify_build.py

Expected: **438 passed**.

---

## What failed

    assert ntpath.isabs("/abs/path") is False
    AssertionError: assert True is False

Not a platform difference this time — a **Python version** difference:

    my sandbox   Python 3.13.14  ->  ntpath.isabs("/abs/path") = False
    your machine Python 3.12.10  ->  ntpath.isabs("/abs/path") = True

CPython changed this in 3.13 (gh-104614): a single leading slash is no longer
reported absolute by `ntpath`. I wrote the assertion on 3.13 and it encoded
3.13's behaviour.

## Why I deleted it rather than fixing the constant

That test asserted a fact about **CPython**, not about this app. It could
never have caught a bug in our code, and it breaks whenever the standard
library changes. Correcting the expected value would have kept a test whose
only function is to fail on a different Python.

The behaviour that actually matters — that `_is_absolute()` treats drive
letters as absolute on every platform and every Python version — is already
covered, and I added one more that exercises the real anchoring code with a
Windows path:

    test_anchoring_never_mangles_a_windows_path

That one would catch the original bug (`/home/user/D:/MyData`) on any
platform, because it runs our code rather than asserting someone else's.

I also swept every test file for assertions on `ntpath`/`posixpath`: none
remain.

---

## Tests: 438

Net zero from last time: one worthless test removed, one useful one added.

---

## The pattern, three rounds running

1. Path anchoring: app correct, my test used forward slashes
2. Absolute paths: app correct, my test used POSIX semantics
3. This one: app correct, my test used 3.13 stdlib behaviour

**The app code has been right every single time.** Each failure was my test
encoding an assumption from the environment I write in rather than the one you
run in.

The lesson I should have applied two rounds ago: a test that asserts anything
about the platform or the standard library is testing the wrong thing. Tests
should exercise our code and assert on its output.

---

## Now finish the build

The two real fixes — `app_root()` and `llama_cpp/lib` — have been sitting
unapplied through all of this because the suite kept failing before the
installer would write them.

1. `install_patch.py` → **438 passed**
2. `build_exe.py` → 5–15 minutes; copies `config.json`, `models\` and `data\`
   beside the exe automatically and fails if `llama_cpp\lib` is missing
3. `verify_build.py` → confirms the exe is current, not stale
4. Launch `dist\AICompanion\AICompanion.exe`

Paste anything that fails at step 2 or 3.
