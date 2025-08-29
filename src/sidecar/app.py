# Auto-discover libmpv.dylib on macOS and set MPV_LIBRARY_PATH if needed
import os, sys, subprocess
from glob import glob
import platform, ctypes

def _ensure_libmpv_env():
    # If user already exported a path, normalize + ensure DYLD paths + preload
    env_path = os.environ.get("MPV_LIBRARY") or os.environ.get("MPV_LIBRARY_PATH")
    if env_path and os.path.exists(env_path):
        _set_lib_env(env_path)
        print(f"[sidecar] Respecting exported MPV lib: {env_path} (arch={platform.machine()})")
        _preload_ctypes(env_path)
        return

    # Fast-path: Homebrew common symlink
    hb_default = "/opt/homebrew/lib/libmpv.dylib"
    if os.path.exists(hb_default):
        _set_lib_env(hb_default)
        print(f"[sidecar] Using Homebrew lib: {hb_default} (arch={platform.machine()})")
        _preload_ctypes(hb_default)
        return

    search_dirs = []
    try:
        opt = subprocess.check_output(["brew", "--prefix", "mpv"], text=True).strip()
        if opt:
            search_dirs.append(os.path.join(opt, "lib"))
    except Exception:
        pass
    try:
        hb = subprocess.check_output(["brew", "--prefix"], text=True).strip()
        if hb:
            search_dirs.append(os.path.join(hb, "lib"))
    except Exception:
        pass
    search_dirs += [
        "/opt/homebrew/lib",
        "/usr/local/lib",
        "/usr/lib",
    ]

    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        exact = os.path.join(d, "libmpv.dylib")
        if os.path.exists(exact):
            candidates.append(exact)
        for p in sorted(glob(os.path.join(d, "libmpv*.dylib"))):
            if p not in candidates:
                candidates.append(p)

    for p in candidates:
        if os.path.exists(p):
            _set_lib_env(p)
            print(f"[sidecar] MPV library set to: {p} (arch={platform.machine()})")
            _preload_ctypes(p)
            return
    # If we reach here, keep env unset; mpv.py will raise a helpful OSError

def _set_lib_env(path: str):
    os.environ["MPV_LIBRARY_PATH"] = path
    os.environ["MPV_LIBRARY"] = path
    libdir = os.path.dirname(path)
    # Prepend to DYLD paths so ctypes.find_library can locate it
    for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        cur = os.environ.get(key, "")
        parts = [libdir] + ([cur] if cur else [])
        os.environ[key] = ":".join([p for p in parts if p])

def _preload_ctypes(path: str):
    try:
        ctypes.CDLL(path)
        print(f"[sidecar] ctypes preloaded: {path}")
    except OSError as e:
        print(f"[sidecar] ctypes.CDLL failed for {path}: {e}")

_ensure_libmpv_env()

from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtCore import QObject, Slot, Signal, QUrl

try:
    import mpv
except OSError as e:
    # Extra diagnostics to help with arch/path issues
    print("[sidecar] mpv import failed:", e)
    print("[sidecar] Env MPV_LIBRARY     =", os.environ.get("MPV_LIBRARY"))
    print("[sidecar] Env MPV_LIBRARY_PATH=", os.environ.get("MPV_LIBRARY_PATH"))
    print("[sidecar] Env DYLD_LIBRARY_PATH=", os.environ.get("DYLD_LIBRARY_PATH"))
    print("[sidecar] Env DYLD_FALLBACK_LIBRARY_PATH=", os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"))
    print("[sidecar] Platform:", platform.platform(), "machine:", platform.machine())
    raise

class MpvWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = mpv.MPV(
            wid=int(self.winId()),
            osc=False, input_default_bindings=False,
            video_sync='display-resample', hr_seek=True
        )
    def load(self, path): self.player.loadfile(path, mode='replace')
    def arm(self, a, b):
        self.player.set_property('ab-loop-a', 'no')
        self.player.set_property('ab-loop-b', 'no')
        self.player.set_property('ab-loop-a', float(a))
        self.player.set_property('ab-loop-b', float(b))
        self.player.seek(float(a), reference='absolute+exact')
        self.player.pause = False
    def mirror(self, on): self.player.vf = 'hflip' if on else ''
    def seek_rel(self, r, a, b): self.player.seek(a + (b-a)*r, reference='absolute+exact')

class Bridge(QObject):
    def __init__(self, clip: MpvWidget, mov: MpvWidget):
        super().__init__()
        self.clip, self.mov = clip, mov
        self.clip_range = (0.0, 0.0)
        self.mov_range  = (0.0, 0.0)

    @Slot(str, str)
    def openFiles(self, clipPath, movPath):
        self.clip.load(clipPath); self.mov.load(movPath)

    @Slot(float, float, float, float, bool, bool)
    def playPair(self, cs, ce, ms, me, mirror, loop):
        self.clip.mirror(mirror)
        self.clip_range = (cs, ce)
        self.mov_range  = (ms, me)
        self.clip.arm(cs, ce)
        self.mov.arm(ms, me)

    @Slot(float)
    def seekRel(self, ratio):
        r = max(0.0, min(1.0, float(ratio)))
        a,b = self.clip_range; self.clip.seek_rel(r, a,b)
        a,b = self.mov_range;  self.mov.seek_rel(r, a,b)

def main():
    app = QApplication([])
    root = QWidget()
    root.setWindowTitle("Recmatcher – Sidecar")
    h = QHBoxLayout(root)
    clip = MpvWidget(); mov = MpvWidget()
    h.addWidget(clip, 1); h.addWidget(mov, 1)

    web = QWebEngineView()
    ch  = QWebChannel()
    br  = Bridge(clip, mov)
    ch.registerObject('pyBridge', br)
    web.page().setWebChannel(ch)
    web.setUrl(QUrl("http://localhost:5173"))  # dev；发布时换成本地文件
    v = QVBoxLayout()
    v.addWidget(web, 1)
    h.addLayout(v, 1)

    root.resize(1600, 900)
    root.show()
    app.exec()

if __name__ == '__main__':
    main()