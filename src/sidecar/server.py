import os, json, socket, subprocess, time, threading
from typing import Optional, Tuple
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====== 配置 ======
MPV_BIN = os.environ.get("MPV_BIN", "mpv")  # 如需指定: export MPV_BIN=/usr/local/bin/mpv
SOCK_CLIP = "/tmp/mpv-clip.sock"
SOCK_MOV  = "/tmp/mpv-movie.sock"

MPV_ARGS_BASE = [
    "--no-terminal",
    "--force-window=yes",
    "--idle=yes",           # 允许启动后等待加载
    "--pause",              # 启动时先暂停，防止抖动
    "--hr-seek=yes",
    "--hr-seek-framedrop=no",
    "--video-sync=display-resample",
    "--demuxer-seekable-cache=yes",
    "--keep-open=no",
    "--profile=low-latency",
    "--osc=no",
    "--no-input-default-bindings",
]

# ====== MPV JSON IPC 工具 ======
class MpvProc:
    def __init__(self, sock_path: str):
        self.sock_path = sock_path
        self.proc: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()

    def start(self):
        # 清理旧 sock
        if os.path.exists(self.sock_path):
            try: os.remove(self.sock_path)
            except: pass
        # 启动 mpv
        cmd = [MPV_BIN, f"--input-ipc-server={self.sock_path}", *MPV_ARGS_BASE]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 等待 socket 就绪
        deadline = time.time() + 5
        while time.time() < deadline:
            if os.path.exists(self.sock_path): break
            time.sleep(0.05)
        # 连接
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.sock_path)

    def ensure(self):
        if not self.proc or self.proc.poll() is not None:
            self.start()
        if not self.sock:
            self.start()

    def _send_cmd(self, cmd):
        data = (json.dumps(cmd) + "\n").encode("utf-8")
        assert self.sock is not None
        self.sock.sendall(data)
        # 简单读一行响应（忽略内容，只要不异常）
        try:
            self.sock.settimeout(0.5)
            self.sock.recv(4096)
        except Exception:
            pass

    def set_prop(self, name, value):
        with self.lock:
            self.ensure()
            self._send_cmd({"command": ["set_property", name, value]})

    def get_prop(self, name):
        with self.lock:
            self.ensure()
            req = {"command": ["get_property", name], "request_id": 1}
            data = (json.dumps(req) + "\n").encode("utf-8")
            self.sock.sendall(data)
            self.sock.settimeout(0.5)
            try:
                buf = self.sock.recv(65536)
                j = json.loads(buf.decode("utf-8", errors="ignore"))
                return j.get("data", None)
            except Exception:
                return None

    def cmd(self, *args):
        with self.lock:
            self.ensure()
            self._send_cmd({"command": list(args)})

    def load_file(self, path: str):
        self.cmd("loadfile", path, "replace")

    def seek_absolute(self, t: float):
        # mpv: seek <value> [relative|absolute|absolute-percent|relative-percent] [exact|keyframes]
        self.cmd("seek", float(t), "absolute+exact")

# ====== 播放控制辅助 ======
def arm_segment(player: MpvProc, start: float, end: float):
    """清理旧 AB、精确 seek 到 start，并开启 [start, end] 的 AB 循环与播放。"""
    s, e = float(start), float(end)
    # 清理旧 AB
    player.set_prop("ab-loop-a", "no")
    player.set_prop("ab-loop-b", "no")
    # 设定新 AB
    player.set_prop("ab-loop-a", s)
    player.set_prop("ab-loop-b", e)
    # 精确回到 A 点
    player.set_prop("hr-seek", True)
    player.seek_absolute(s)
    # 循环与播放
    player.set_prop("loop-file", "inf")
    player.set_prop("pause", False)

def set_mirror(player: MpvProc, enabled: bool):
    """设置/清空水平镜像滤镜。"""
    if enabled:
        player.set_prop("vf", "hflip")
    else:
        # 清空所有 vf（只为简单起见）
        player.set_prop("vf", "")

# ====== 全局状态 ======
clip = MpvProc(SOCK_CLIP)
mov  = MpvProc(SOCK_MOV)

STATE = {
    "movie_path": None,
    "clip_path": None,
    # 最近一次 play_pair 的 AB 范围
    "clip_range": (0.0, 0.0),
    "mov_range":  (0.0, 0.0),
    "loop": True,
    "mirror": False,
}

# ====== FastAPI ======
app = FastAPI(title="recmatcher-sidecar", version="0.1.0")

# Enable CORS for dev (allow all). If you want to restrict, set SIDECAR_ALLOW_ORIGINS env to CSV.
allow_origins_env = os.environ.get("SIDECAR_ALLOW_ORIGINS")
if allow_origins_env:
    allow_origins = [s.strip() for s in allow_origins_env.split(",") if s.strip()]
else:
    allow_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],   # include OPTIONS for preflight
    allow_headers=["*"],
    expose_headers=["*"],
)

class OpenReq(BaseModel):
    movie: str
    clip: str
    minimized: Optional[bool] = True  # 是否最小化窗口（默认最小化，避免打扰）

@app.get("/health")
def health():
    return {"ok": True, "mpv": True}

@app.post("/open")
def open_files(req: OpenReq):
    STATE["movie_path"] = req.movie
    STATE["clip_path"]  = req.clip
    # 启动/连接
    clip.start()
    mov.start()
    # 加载文件
    if os.path.exists(req.clip):
        clip.load_file(req.clip)
    if os.path.exists(req.movie):
        mov.load_file(req.movie)
    # 初始暂停
    clip.set_prop("pause", True)
    mov.set_prop("pause", True)
    # 默认最小化，除非显式要求不最小化
    try:
        minimized = bool(req.minimized)
    except Exception:
        minimized = True
    clip.set_prop("window-minimized", minimized)
    mov.set_prop("window-minimized", minimized)
    return {"ok": True, "paths": {"clip": STATE["clip_path"], "movie": STATE["movie_path"]}, "minimized": minimized}

class PlayPairReq(BaseModel):
    clipStart: float
    clipEnd: float
    movieStart: float
    movieEnd: float
    loop: bool = True
    mirror: bool = False

@app.post("/play_pair")
def play_pair(req: PlayPairReq):
    # 确保进程与文件已打开
    clip.ensure(); mov.ensure()
    if not STATE["clip_path"] or not STATE["movie_path"]:
        return {"ok": False, "error": "files not opened yet. call /open first"}

    # 镜像（仅 clip）
    set_mirror(clip, bool(req.mirror))

    # 记录范围
    a, b = float(req.clipStart), float(req.clipEnd)
    a2, b2 = float(req.movieStart), float(req.movieEnd)
    STATE["clip_range"] = (a, b)
    STATE["mov_range"]  = (a2, b2)
    STATE["loop"] = bool(req.loop)
    STATE["mirror"] = bool(req.mirror)

    # 装填并开始循环
    arm_segment(clip, a, b)
    arm_segment(mov,  a2, b2)
    return {"ok": True}

class WindowReq(BaseModel):
    minimized: bool

@app.post("/window/minimize")
def window_minimize(req: WindowReq):
    clip.set_prop("window-minimized", bool(req.minimized))
    mov.set_prop("window-minimized", bool(req.minimized))
    return {"ok": True}

class LayoutReq(BaseModel):
    # 比例（例如 0.5 = 占屏幕一半宽度/高度）
    scale: float = 0.5
    gap: int = 10
    # 屏幕分辨率（macOS 多屏可手动指定）
    screen_w: int = 1920
    screen_h: int = 1080

@app.post("/window/layout")
def window_layout(req: LayoutReq):
    w = int(req.screen_w * req.scale)
    h = int(req.screen_h * req.scale)
    gap = int(req.gap)
    # 左右平铺
    clip.set_prop("geometry", f"{w}x{h}+{gap}+{gap}")
    mov.set_prop("geometry",  f"{w}x{h}+{w + 2*gap}+{gap}")
    # 取消最小化，置于最前方便观察
    clip.set_prop("window-minimized", False)
    mov.set_prop("window-minimized", False)
    clip.set_prop("ontop", True)
    mov.set_prop("ontop", True)
    return {"ok": True, "geometry": {"w": w, "h": h}}

@app.post("/pause")
def pause():
    clip.set_prop("pause", True)
    mov.set_prop("pause", True)
    return {"ok": True}

@app.post("/resume")
def resume():
    clip.set_prop("pause", False)
    mov.set_prop("pause", False)
    return {"ok": True}

class SeekRelReq(BaseModel):
    ratio: float  # 0~1

@app.post("/seek_rel")
def seek_rel(req: SeekRelReq):
    r = max(0.0, min(1.0, float(req.ratio)))
    c_a, c_b = STATE["clip_range"]
    m_a, m_b = STATE["mov_range"]
    c_t = c_a + (c_b - c_a) * r
    m_t = m_a + (m_b - m_a) * r
    clip.seek_absolute(c_t)
    mov.seek_absolute(m_t)
    return {"ok": True}

@app.get("/status")
def status():
    # 读取 mpv 当前时间
    def read_state(p: MpvProc):
        t = p.get_prop("time-pos")
        paused = p.get_prop("pause")
        # 规范化
        if isinstance(paused, (int, float)):
            paused = bool(paused)
        return t, bool(paused) if paused is not None else None

    ct, paused_c = read_state(clip)
    mt, paused_m = read_state(mov)
    return {
        "ok": True,
        "clip": {"t": ct, "paused": paused_c},
        "movie": {"t": mt, "paused": paused_m},
        "ranges": {
            "clip": STATE["clip_range"],
            "movie": STATE["mov_range"],
        },
        "loop": STATE["loop"],
        "mirror": STATE["mirror"],
        "paths": {"clip": STATE["clip_path"], "movie": STATE["movie_path"]},
        "playing": (paused_c is False) and (paused_m is False),
    }