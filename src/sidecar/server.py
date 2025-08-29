import os, json, socket, subprocess, time, threading
from typing import Optional, Tuple
from fastapi import FastAPI, Body
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

class OpenReq(BaseModel):
    movie: str
    clip: str

@app.get("/health")
def health():
    return {"ok": True, "mpv": True}

@app.post("/open")
def open_files(req: OpenReq):
    STATE["movie_path"] = req.movie
    STATE["clip_path"]  = req.clip
    # 启动/加载
    clip.start()
    mov.start()
    if os.path.exists(req.clip):
        clip.load_file(req.clip)
    if os.path.exists(req.movie):
        mov.load_file(req.movie)
    # 初始为暂停
    clip.set_prop("pause", True)
    mov.set_prop("pause", True)
    return {"ok": True}

class PlayPairReq(BaseModel):
    clipStart: float
    clipEnd: float
    movieStart: float
    movieEnd: float
    loop: bool = True
    mirror: bool = False

@app.post("/play_pair")
def play_pair(req: PlayPairReq):
    # 确保已加载
    clip.ensure(); mov.ensure()
    # 再次 load（确保当前路径正确）
    if STATE["clip_path"]:
        clip.load_file(STATE["clip_path"])
    if STATE["movie_path"]:
        mov.load_file(STATE["movie_path"])

    # 设置 AB 循环
    a, b = float(req.clipStart), float(req.clipEnd)
    clip.set_prop("ab-loop-a", a)
    clip.set_prop("ab-loop-b", b)

    a2, b2 = float(req.movieStart), float(req.movieEnd)
    mov.set_prop("ab-loop-a", a2)
    mov.set_prop("ab-loop-b", b2)

    # 镜像仅对 clip
    if req.mirror:
        clip.set_prop("vf", "hflip")
    else:
        clip.set_prop("vf", "")

    # 从 A 精确起播
    clip.seek_absolute(a)
    mov.seek_absolute(a2)

    # 循环与播放
    STATE["clip_range"] = (a, b)
    STATE["mov_range"]  = (a2, b2)
    STATE["loop"] = bool(req.loop)
    STATE["mirror"] = bool(req.mirror)

    # mpv 的 ab-loop 会自动回绕；是否强制 loop 次数由 mpv 管
    clip.set_prop("pause", False)
    mov.set_prop("pause", False)

    return {"ok": True}

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
    ct = clip.get_prop("playback-time")
    mt = mov.get_prop("playback-time")
    paused_c = clip.get_prop("pause")
    paused_m = mov.get_prop("pause")
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
    }