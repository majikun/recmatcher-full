#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Qt WebEngine launcher for recmatcher UI
- 首要目标：把网页+两个视频窗口整到一个 Qt App 里
- 不依赖浏览器 <video>，为后续 mpv 内嵌做铺垫
- 提供足量日志，优雅降级，不因环境缺失而崩溃
"""

import os
import sys
import argparse

# ------- 在导入 Qt 之前强制使用 OpenGL（避免 macOS 默认 Metal 与 QOpenGLWidget 冲突） -------
os.environ.setdefault("QT_OPENGL", "desktop")        # 强制使用 OpenGL
os.environ.setdefault("QT_WIDGETS_RHI", "0")         # 关闭 Widgets 的 RHI/Metal
os.environ.setdefault("QTWEBENGINE_REMOTE_DEBUGGING", "9223")
os.environ.setdefault("QT_LOGGING_RULES", "qt.webengine.*=true;qt.network.ssl.warning=true")

DEFAULT_URL = os.environ.get("SIDECAR_URL", "http://localhost:5173")
DEVTOOLS = os.environ.get("SIDECAR_DEVTOOLS", "0") == "1"

# ------- 尝试预加载 libmpv（尊重用户提供的 MPV_LIBRARY/MPV_LIBRARY_PATH） -------
MPV_HINT = os.environ.get("MPV_LIBRARY") or os.environ.get("MPV_LIBRARY_PATH")
if MPV_HINT and os.path.exists(MPV_HINT):
    try:
        import ctypes
        ctypes.CDLL(MPV_HINT)  # 预加载，帮助 python-mpv 找到 libmpv
        print(f"[minweb] preloaded libmpv: {MPV_HINT}")
    except Exception as e:
        print(f"[minweb] failed to preload libmpv from {MPV_HINT}: {e}")
else:
    if MPV_HINT:
        print(f"[minweb] MPV hint path not found: {MPV_HINT}")
    else:
        print("[minweb] no MPV_LIBRARY / MPV_LIBRARY_PATH was provided")

# ------- 之后再导入 Qt 模块 -------
from PySide6 import QtCore, QtGui, QtWidgets, QtWebEngineCore, QtWebEngineWidgets

# QOpenGLWidget 位于 QtOpenGLWidgets（不是 QtWidgets）
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    HAVE_QOPENGLWIDGET = True
except Exception as e:
    QOpenGLWidget = None
    HAVE_QOPENGLWIDGET = False
    print("[minweb] QtOpenGLWidgets import failed:", e)

# 尝试导入 python-mpv（不强求 PyOpenGL：mpv 的 opengl-cb 不需要 PyOpenGL 也能跑）
try:
    import mpv  # python-mpv (>=1.0.6 recommended)
    HAVE_MPV = True
except Exception as e:
    mpv = None
    HAVE_MPV = False
    print("[minweb] mpv import failed:", e)


class LoggingWebPage(QtWebEngineCore.QWebEnginePage):
    """把页面内的 console.* 输出到终端，便于排查空白/报错"""
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        lv = {0: "INFO", 1: "WARN", 2: "ERR"}.get(level, str(level))
        print(f"[minweb][JS {lv}] {sourceID}:{lineNumber} {message}")

    def certificateError(self, error):
        print(f"[minweb] certificateError: {error.errorDescription()} (ignored for dev)")
        return True

    def renderProcessTerminated(self, status, exitCode):
        print(f"[minweb] renderProcessTerminated: status={status} exitCode={exitCode}")
        super().renderProcessTerminated(status, exitCode)


class MpvGLWidget(QOpenGLWidget if HAVE_QOPENGLWIDGET else QtWidgets.QWidget):
    """
    Minimal libmpv opengl-cb widget.
    - 若缺少 libmpv 或 QOpenGLWidget，则降级为普通 QWidget，显示提示文本，不崩溃。
    - 不依赖 PyOpenGL；仅使用 Qt 提供的 OpenGL 上下文给 mpv。
    """
    def __init__(self, parent=None, title="mpv"):
        super().__init__(parent)
        self.setObjectName(f"MpvGLWidget[{title}]")
        self._title = title
        # 只要求 mpv + QOpenGLWidget（不再强制 PyOpenGL）
        self._ok = bool(HAVE_MPV and HAVE_QOPENGLWIDGET)
        self._mpv = None
        self._w = 0
        self._h = 0
        self._pending_file = None

        if not self._ok:
            # Fallback: 显示缺失原因
            self._label = QtWidgets.QLabel(self)
            self._label.setAlignment(QtCore.Qt.AlignCenter)
            miss = []
            if not HAVE_MPV: miss.append("mpv")
            if not HAVE_QOPENGLWIDGET: miss.append("QOpenGLWidget")
            self._label.setText("Missing: " + ", ".join(miss))
            self._label.setStyleSheet("color:#c00; font: 13px 'Menlo';")
            self._label.show()
            self.setMinimumSize(320, 180)
        else:
            self._label = None
        print(f"[minweb][{self._title}] widget init, ok={self._ok} (mpv={HAVE_MPV}, qglw={HAVE_QOPENGLWIDGET})")

    # ---- libmpv (OpenGL-cb) glue ----
    def _get_proc_addr(self, name: bytes):
        try:
            func = self.context().getProcAddress(name.decode("utf-8"))
            return int(func) if func else 0
        except Exception:
            return 0

    def initializeGL(self):
        if not HAVE_QOPENGLWIDGET:
            return
        self._w = max(1, self.width())
        self._h = max(1, self.height())
        print(f"[minweb][{self._title}] initializeGL w={self._w} h={self._h}")
        if not self._ok:
            print(f"[minweb][{self._title}] initializeGL skipped: ok={self._ok} (mpv={HAVE_MPV}, qglw={HAVE_QOPENGLWIDGET})")
            return
        try:
            # 创建 mpv（opengl-cb 模式，不弹出独立窗口）
            self._mpv = mpv.MPV(
                ytdl=False,
                config=False,
                vid="auto",
                osc=False,
                input_default_bindings=False,
                log_handler=lambda l, m: print(f"[mpv:{self._title}][{l}] {m}"),
                opengl_cb=True,
            )
            self._mpv["vo"] = "gpu"
            self._mpv["gpu-api"] = "opengl"
            if hasattr(self._mpv, "opengl_cb_init_gl"):
                self._mpv.opengl_cb_init_gl(self._get_proc_addr)
            if hasattr(self._mpv, "opengl_cb_set_update_callback"):
                self._mpv.opengl_cb_set_update_callback(self._on_mpv_update)

            if self._pending_file:
                self.load(self._pending_file)
        except Exception as e:
            print(f"[minweb][{self._title}] mpv init failed:", e)
            self._ok = False

    def resizeGL(self, w, h):
        if not HAVE_QOPENGLWIDGET:
            return
        self._w, self._h = max(1, w), max(1, h)
        self.update()

    def paintGL(self):
        if not HAVE_QOPENGLWIDGET:
            return
        if not (self._ok and self._mpv):
            # 无帧可绘制时，保持当前 FBO 内容
            return
        try:
            fbo = int(self.defaultFramebufferObject())
            if hasattr(self._mpv, "opengl_cb_draw"):
                # 负高度以适配 Qt FBO 的坐标系
                self._mpv.opengl_cb_draw(fbo, self._w, -self._h)
        except Exception as e:
            print(f"[minweb][{self._title}] paintGL error:", e)

    def _on_mpv_update(self):
        self.update()

    def load(self, path: str):
        if not self._ok:
            print(f"[minweb][{self._title}] load skipped (mpv/QOpenGLWidget not ready): {path}")
            self._pending_file = path
            return
        try:
            print(f"[minweb][{self._title}] loadfile: {path}")
            self._mpv.command("loadfile", path, "replace")
            self._mpv["loop-file"] = "inf"
            self._mpv["hwdec"] = "auto-safe"
            self._mpv["keep-open"] = "no"
            self._mpv["mute"] = "yes"
        except Exception as e:
            print(f"[minweb][{self._title}] load failed:", e)

    def closeEvent(self, ev):
        try:
            if self._mpv:
                self._mpv.terminate()
        except Exception:
            pass
        super().closeEvent(ev)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, url: str):
        super().__init__()
        self.setWindowTitle("Recmatcher – Minimal Web Shell")
        self.resize(1280, 800)

        # 左侧：两个 mpv 竖直堆叠
        self.clipView = MpvGLWidget(self, title="clip")
        self.movieView = MpvGLWidget(self, title="movie")
        vleft = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        vleft.addWidget(self.clipView)
        vleft.addWidget(self.movieView)
        vleft.setSizes([1, 1])

        # 右侧：网页
        self.view = QtWebEngineWidgets.QWebEngineView(self)
        self.page = LoggingWebPage(self.view)
        self.view.setPage(self.page)

        # 顶层：水平分割
        hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        hsplit.addWidget(vleft)
        hsplit.addWidget(self.view)
        hsplit.setStretchFactor(0, 1)
        hsplit.setStretchFactor(1, 1)

        central = QtWidgets.QWidget(self)
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(hsplit)
        self.setCentralWidget(central)

        # 基础设置
        prof: QtWebEngineCore.QWebEngineProfile = self.page.profile()
        prof.setHttpCacheType(QtWebEngineCore.QWebEngineProfile.HttpCacheType.DiskHttpCache)
        prof.setPersistentCookiesPolicy(QtWebEngineCore.QWebEngineProfile.AllowPersistentCookies)
        prof.setSpellCheckEnabled(False)

        s: QtWebEngineCore.QWebEngineSettings = self.page.settings()
        s.setAttribute(QtWebEngineCore.QWebEngineSettings.JavascriptEnabled, True)
        s.setAttribute(QtWebEngineCore.QWebEngineSettings.LocalStorageEnabled, True)
        s.setAttribute(QtWebEngineCore.QWebEngineSettings.ScrollAnimatorEnabled, True)
        s.setAttribute(QtWebEngineCore.QWebEngineSettings.ErrorPageEnabled, True)
        s.setAttribute(QtWebEngineCore.QWebEngineSettings.FullScreenSupportEnabled, True)
        s.setAttribute(QtWebEngineCore.QWebEngineSettings.JavascriptCanOpenWindows, True)

        # 顶部工具栏：地址栏 + Go + Reload + DevTools + 载入测试视频
        toolbar = QtWidgets.QToolBar("Navigation", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.addr = QtWidgets.QLineEdit(url, self)
        self.addr.returnPressed.connect(self.load_from_bar)
        toolbar.addWidget(self.addr)

        go_act = QtGui.QAction("Go", self)
        go_act.triggered.connect(self.load_from_bar)
        toolbar.addAction(go_act)

        reload_act = QtGui.QAction("Reload", self)
        reload_act.triggered.connect(self.view.reload)
        toolbar.addAction(reload_act)

        if DEVTOOLS:
            devtools_act = QtGui.QAction("DevTools", self)
            devtools_act.triggered.connect(self.open_devtools)
            toolbar.addAction(devtools_act)

        load_test_act = QtGui.QAction("Load Test Videos", self)
        def _load_test():
            clip_path = "/opt/homebrew/var/www/videos/output/20250805/V100784_MacMax_cropped/smart_reclip/clip_scaled.mp4"
            movie_path = "/opt/homebrew/var/www/videos/movie_assets/the_help/movie.mp4"
            self.clipView.load(clip_path)
            self.movieView.load(movie_path)
        load_test_act.triggered.connect(_load_test)
        toolbar.addAction(load_test_act)

        # 连接加载信号，打印详细日志
        self.view.loadStarted.connect(lambda: print("[minweb] loadStarted"))
        self.view.loadProgress.connect(lambda p: print(f"[minweb] loadProgress: {p}%"))
        self.view.loadFinished.connect(lambda ok: print(f"[minweb] loadFinished: ok={ok}"))
        self.view.urlChanged.connect(lambda u: print(f"[minweb] urlChanged: {u.toString()}"))

        # 初始加载
        self.load_url(url)

        # 启动后自动试播一次（可注释）
        QtCore.QTimer.singleShot(800, _load_test)

    def load_from_bar(self):
        self.load_url(self.addr.text().strip() or "about:blank")

    def load_url(self, url: str):
        try:
            qurl = QtCore.QUrl(url)
            if not qurl.isValid():
                print(f"[minweb] Invalid URL: {url}")
                return
            print(f"[minweb] loading: {url}")
            self.view.setUrl(qurl)
        except Exception as e:
            print(f"[minweb] load_url error: {e}")

    def open_devtools(self):
        print("[minweb] DevTools requested")
        try:
            if hasattr(self.page, "devToolsPage") and self.page.devToolsPage() is not None:
                dev = QtWebEngineWidgets.QWebEngineView(self)
                dev.setWindowTitle("DevTools")
                dev.setPage(self.page.devToolsPage())
                dev.resize(960, 700)
                dev.show()
                print("[minweb] DevTools opened in embedded view")
            else:
                import webbrowser
                port = os.environ.get("QTWEBENGINE_REMOTE_DEBUGGING", "9223")
                url = f"http://127.0.0.1:{port}"
                webbrowser.open(url)
                print(f"[minweb] DevTools not embedded; opened external {url}")
        except Exception as e:
            print(f"[minweb] DevTools open failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Minimal Web Shell (Qt WebEngine) for recmatcher UI")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"URL to load (default: {DEFAULT_URL})")
    args = parser.parse_args()

    # 在 QApplication 之前，指定 OpenGL Profile（进一步确保不是 Metal）
    try:
        fmt = QtGui.QSurfaceFormat()
        fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
        fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        fmt.setVersion(3, 2)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        print("[minweb] set OpenGL default surface format: Core 3.2")
    except Exception as e:
        print("[minweb] setDefaultFormat failed:", e)

    print("[minweb] PySide6 version:", QtCore.qVersion())
    print("[minweb] QtWebEngineCore:", getattr(QtWebEngineCore, "__file__", "n/a"))
    print("[minweb] QtWebEngineWidgets:", getattr(QtWebEngineWidgets, "__file__", "n/a"))
    print("[minweb] Env QTWEBENGINE_REMOTE_DEBUGGING =", os.environ.get("QTWEBENGINE_REMOTE_DEBUGGING"))
    print("[minweb] Env SIDECAR_DEVTOOLS =", os.environ.get("SIDECAR_DEVTOOLS"))
    print("[minweb] Env QT_OPENGL =", os.environ.get("QT_OPENGL"))
    print("[minweb] Env QT_WIDGETS_RHI =", os.environ.get("QT_WIDGETS_RHI"))
    print("[minweb] Env MPV_LIBRARY =", os.environ.get("MPV_LIBRARY"))
    print("[minweb] Env MPV_LIBRARY_PATH =", os.environ.get("MPV_LIBRARY_PATH"))

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args.url)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()