import subprocess
from pathlib import Path
import sys

stubs_dir = Path("/home/user/qtstubs")
stubs_dir.mkdir(exist_ok=True)

# Find site-packages containing PySide6
site = None
for p in sys.path:
    pp = Path(p)
    if (pp / "PySide6").is_dir():
        site = pp
        break
if site is None:
    raise RuntimeError("PySide6 not found in sys.path")

def get_syms(prefix):
    syms = set()
    for so in site.glob("**/*.so*"):
        try:
            out = subprocess.run(["readelf", "-s", "--wide", str(so)], capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split()
                if "UND" in parts:
                    idx = parts.index("UND")
                    if idx + 1 < len(parts):
                        sym = parts[idx + 1].split("@")[0]
                        if sym.startswith(prefix):
                            syms.add(sym)
        except Exception:
            pass
    return sorted(syms)

# libGL
gl_syms = get_syms("gl")
c_gl = "\n".join(f"void {s}(void) {{}}" for s in gl_syms)
(stubs_dir / "libGL.c").write_text(c_gl)
subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(stubs_dir / "libGL.so.1"), str(stubs_dir / "libGL.c")], check=True)

# libEGL
egl_syms = get_syms("egl")
c_egl = "\n".join(f"void {s}(void) {{}}" for s in egl_syms)
(stubs_dir / "libEGL.c").write_text(c_egl)
subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(stubs_dir / "libEGL.so.1"), str(stubs_dir / "libEGL.c")], check=True)

# libxkbcommon
xkb_syms = get_syms("xkb_")
c_xkb = "\n".join(f"void {s}(void) {{}}" for s in xkb_syms)
(stubs_dir / "libxkb.c").write_text(c_xkb)
(stubs_dir / "xkb.ver").write_text("V_0.5.0 { global: *; };\n")
subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,--version-script=" + str(stubs_dir / "xkb.ver"), "-o", str(stubs_dir / "libxkbcommon.so.0"), str(stubs_dir / "libxkb.c")], check=True)

# libdbus-1
dbus_syms = get_syms("dbus_")
c_dbus = "\n".join(f"void {s}(void) {{}}" for s in dbus_syms)
(stubs_dir / "libdbus.c").write_text(c_dbus)
(stubs_dir / "dbus.ver").write_text("LIBDBUS_1_3 { global: *; };\n")
subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,--version-script=" + str(stubs_dir / "dbus.ver"), "-o", str(stubs_dir / "libdbus-1.so.3"), str(stubs_dir / "libdbus.c")], check=True)

print("Generated qtstubs:", len(gl_syms), len(egl_syms), len(xkb_syms), len(dbus_syms))
