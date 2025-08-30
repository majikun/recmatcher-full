# Auto-discover libmpv.dylib on macOS and set MPV_LIBRARY_PATH if needed
import os, sys, subprocess
from glob import glob
import platform, ctypes

os.environ.setdefault('QTWEBENGINE_REMOTE_DEBUGGING','9223')
os.environ.setdefault('QT_LOGGING_RULES','qt.webengine.*=true;qt.network.ssl.warning=true')

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
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineScript
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtCore import QObject, Slot, Signal, QUrl

import PySide6
print(f"[sidecar] PySide6 version: {PySide6.__version__}")

def maybe_init_webengine():
    """
    Try to initialize Qt WebEngine **only if** the symbol exists.
    On PySide6 6.7/6.8 the static function may or may not be present; on 6.9 it is removed.
    """
    try:
        from PySide6.QtWebEngineCore import QWebEngine  # type: ignore
        try:
            QWebEngine.initialize()
            print("[sidecar] QWebEngine.initialize() OK")
        except Exception as e:
            print("[sidecar] QWebEngine.initialize() present but call failed (usually harmless):", e)
    except Exception:
        print("[sidecar] QWebEngine symbol not available; skipping explicit initialize")

class LoggingWebPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"[sidecar][JS Console][{sourceID}:{lineNumber}] {message}")

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
        print("[sidecar] Initializing MpvWidget...")
        try:
            self.player = mpv.MPV(
                wid=int(self.winId()),
                osc=False, input_default_bindings=False,
                video_sync='display-resample', hr_seek=True
            )
        except Exception as e:
            print(f"[sidecar] Failed to initialize mpv.MPV: {e}")
            raise
        print("[sidecar] MpvWidget initialized with player:", self.player)
    def load(self, path):
        print(f"[sidecar] MpvWidget.load called with path: {path}")
        self.player.loadfile(path, mode='replace')
    def arm(self, a, b):
        print(f"[sidecar] MpvWidget.arm called with a={a}, b={b}")
        self.player.set_property('ab-loop-a', 'no')
        self.player.set_property('ab-loop-b', 'no')
        self.player.set_property('ab-loop-a', float(a))
        self.player.set_property('ab-loop-b', float(b))
        self.player.seek(float(a), reference='absolute+exact')
        self.player.pause = False
    def mirror(self, on):
        print(f"[sidecar] MpvWidget.mirror called with on={on}")
        self.player.vf = 'hflip' if on else ''
    def seek_rel(self, r, a, b):
        print(f"[sidecar] MpvWidget.seek_rel called with r={r}, a={a}, b={b}")
        self.player.seek(a + (b-a)*r, reference='absolute+exact')

class Bridge(QObject):
    def __init__(self, clip: MpvWidget, mov: MpvWidget):
        super().__init__()
        self.clip, self.mov = clip, mov
        self.clip_range = (0.0, 0.0)
        self.mov_range  = (0.0, 0.0)

    @Slot(result=str)
    def ping(self):
        return "pong"

    @Slot(result='QVariant')
    def getStatus(self):
        return {
            "clip_range": list(self.clip_range),
            "movie_range": list(self.mov_range),
        }

    @Slot(str, str)
    def openFiles(self, clipPath, movPath):
        print(f"[sidecar] Bridge.openFiles called with clipPath={clipPath}, movPath={movPath}")
        self.clip.load(clipPath); self.mov.load(movPath)

    @Slot(float, float, float, float, bool, bool)
    def playPair(self, cs, ce, ms, me, mirror, loop):
        print(f"[sidecar] Bridge.playPair called with cs={cs}, ce={ce}, ms={ms}, me={me}, mirror={mirror}, loop={loop}")
        self.clip.mirror(mirror)
        self.clip_range = (cs, ce)
        self.mov_range  = (ms, me)
        self.clip.arm(cs, ce)
        self.mov.arm(ms, me)

    @Slot(float)
    def seekRel(self, ratio):
        print(f"[sidecar] Bridge.seekRel called with ratio={ratio}")
        r = max(0.0, min(1.0, float(ratio)))
        a,b = self.clip_range; self.clip.seek_rel(r, a,b)
        a,b = self.mov_range;  self.mov.seek_rel(r, a,b)

def main():
    app = QApplication([])

    try:
        import PySide6
        from PySide6 import QtWebEngineCore, QtWebEngineWidgets
        print("[sidecar] QtWebEngineCore path:", getattr(QtWebEngineCore, "__file__", "n/a"))
        print("[sidecar] QtWebEngineWidgets path:", getattr(QtWebEngineWidgets, "__file__", "n/a"))
    except Exception as e:
        print("[sidecar] QtWebEngine modules import diagnostic failed:", e)

    # Initialize Qt WebEngine if the symbol exists (tolerant across PySide6 versions)
    maybe_init_webengine()
    root = QWidget()
    root.setWindowTitle("Recmatcher – Sidecar")
    h = QHBoxLayout(root)
    clip = MpvWidget(); mov = MpvWidget()
    h.addWidget(clip, 1); h.addWidget(mov, 1)

    web = QWebEngineView()
    page = LoggingWebPage()
    web.setPage(page)

    ch  = QWebChannel()
    br  = Bridge(clip, mov)
    ch.registerObject('pyBridge', br)
    print("[sidecar] QWebChannel registered object 'pyBridge'")
    page.setWebChannel(ch)

    # Inject a small bootstrap that (1) ensures qwebchannel.js is available, then
    # (2) creates a global `window.sidecar` bound to our Bridge via QWebChannel.
    init_js = r"""
    (function(){
      if (window.__rm_webchannel_init__) return;
      window.__rm_webchannel_init__ = true;

      function attachChannel(){
        try{
          new QWebChannel(qt.webChannelTransport, function(ch){
            window.sidecar = ch.objects.pyBridge;
            window.dispatchEvent(new Event('sidecar-ready'));
            console.log('[sidecar] QWebChannel connected =', !!window.sidecar);
            if (window.sidecar && typeof window.sidecar.ping === 'function') {
              try {
                var r = window.sidecar.ping();
                if (r && typeof r.then === 'function') { r.then(function(x){ console.log('[sidecar] ping ->', x); }); }
              } catch (e) { console.log('[sidecar] ping call error', e); }
            }
          });
        }catch(e){
          console.log('[sidecar] attachChannel() failed', e);
        }
      }

      function ensureQWebChannel(){
        if (typeof QWebChannel === 'function') { attachChannel(); return; }
        var s = document.createElement('script');
        s.src = 'qrc:///qtwebchannel/qwebchannel.js';
        s.onload = attachChannel;
        s.onerror = function(){ console.log('[sidecar] qwebchannel.js load error'); };
        document.head.appendChild(s);
      }

      if (typeof qt !== 'undefined' && qt.webChannelTransport) {
        ensureQWebChannel();
      } else {
        console.log('[sidecar] qt.webChannelTransport not ready yet');
        document.addEventListener('DOMContentLoaded', ensureQWebChannel);
      }
    })();
    """

    scr = QWebEngineScript()
    scr.setName("rm-webchannel-init")
    scr.setWorldId(QWebEngineScript.MainWorld)
    scr.setInjectionPoint(QWebEngineScript.DocumentReady)
    scr.setRunsOnSubFrames(False)
    scr.setSourceCode(init_js)
    page.profile().scripts().insert(scr)

    def on_load_started():
        print("[sidecar] WebEngine load started")

    def on_load_progress(p):
        print(f"[sidecar] WebEngine load progress: {p}%")

    def on_load_finished(ok):
        print(f"[sidecar] WebEngine load finished, success={ok}")
        page.runJavaScript("console.log('[sidecar] window.sidecar present?', !!window.sidecar)")
        if not ok:
            print("[sidecar] ERROR: Failed to load web content. Check the URL or server status.")
        else:
            js_check = "console.log('[sidecar] web ready, has qt? ', typeof window.qt !== 'undefined');"
            page.runJavaScript(js_check)

    web.loadStarted.connect(on_load_started)
    web.loadProgress.connect(on_load_progress)
    web.loadFinished.connect(on_load_finished)

    url = QUrl("http://localhost:5173")
    web.setUrl(url)  # dev；发布时换成本地文件

    v = QVBoxLayout()
    v.addWidget(web, 1)
    h.addLayout(v, 1)

    root.resize(1600, 900)
    root.show()

    print(f"[sidecar] Starting application with URL: {url.toString()}")
    print(f"[sidecar] Environment variables:")
    print(f"  MPV_LIBRARY: {os.environ.get('MPV_LIBRARY')}")
    print(f"  MPV_LIBRARY_PATH: {os.environ.get('MPV_LIBRARY_PATH')}")
    print(f"  DYLD_LIBRARY_PATH: {os.environ.get('DYLD_LIBRARY_PATH')}")
    print(f"  DYLD_FALLBACK_LIBRARY_PATH: {os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')}")
    print(f"  QTWEBENGINE_REMOTE_DEBUGGING: {os.environ.get('QTWEBENGINE_REMOTE_DEBUGGING')}")
    print(f"  SIDECAR_DEVTOOLS: {os.environ.get('SIDECAR_DEVTOOLS')}")

    # Optional DevTools window
    if os.environ.get('SIDECAR_DEVTOOLS') == '1':
        print("[sidecar] SIDECAR_DEVTOOLS=1 detected, opening DevTools window")
        devtools = QWebEngineView()
        devtools.setPage(page.devToolsPage())
        devtools.setWindowTitle("Recmatcher – Sidecar DevTools")
        devtools.resize(800, 600)
        devtools.show()

    app.exec()

if __name__ == '__main__':
    main()