from __future__ import annotations
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
import subprocess, os
from .state import STATE
from .schemas import OpenProjectReq, ApplyBatchReq, SaveReq
from .utils import group_by_clip_scene
from pathlib import Path
import json

# --- overrides sidecar helpers -------------------------------------------------

def _overrides_path() -> Path:
    root = Path(STATE.project_root or ".")
    return root / "match_overrides.json"


def _load_overrides_into_state() -> None:
    """Load sidecar overrides into STATE.applied_changes (id -> chosen dict)."""
    p = _overrides_path()
    applied: dict[int, dict] = {}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        applied[int(k)] = v
                    except Exception:
                        pass
            elif isinstance(raw, list):  # legacy list format: [{"seg_id":.., "chosen":..}]
                for it in raw:
                    sid = it.get("seg_id")
                    if isinstance(sid, int):
                        applied[sid] = it.get("chosen")
        except Exception:
            applied = {}
    STATE.applied_changes = applied


def _save_overrides_from_state() -> Path:
    """Persist STATE.applied_changes to sidecar file. Returns the path."""
    p = _overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {str(k): v for k, v in (STATE.applied_changes or {}).items()}
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p

app = FastAPI(title="Recmatcher UI Backend", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _scenes_summary():
    groups = group_by_clip_scene(STATE.match_segments)
    order = []
    last = object()
    for r in STATE.match_segments:
        cid = r.get("scene_id")
        if cid != last:
            order.append(cid)
            last = cid
    data = []
    for cid in order:
        segs = groups.get(cid, [])
        avg = 0.0
        if segs:
            vals = []
            for s in segs:
                mo = s.get("matched_orig_seg") or {}
                sc = float(mo.get("score") or 0.0)
                if mo.get("is_fill"):
                    sc *= 0.5
                vals.append(sc)
            avg = sum(vals) / len(vals)
        chain_len = 1
        key = str(cid)
        if key in STATE.scene_out:
            chain_len = int(STATE.scene_out[key].get("stats", {}).get("chain_len", 1))
        data.append({
            "clip_scene_id": cid,
            "count": len(segs),
            "avg_conf": avg,
            "chain_len": chain_len,
        })
    return data

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/project/open")
def open_project(req: OpenProjectReq):
    if not Path(req.root).exists():
        raise HTTPException(404, "root not found")
    STATE.load_project(req.root, req.movie_path, req.clip_path)
    STATE.build_explain_offsets()
    _load_overrides_into_state()
    scenes = _scenes_summary()
    return {"ok": True, "scenes": scenes, "paths": STATE.paths}

@app.get("/video/movie")
def video_movie(t: float | None = Query(None)):
    p = STATE.paths.get("movie")
    if not p or not Path(p).exists():
        raise HTTPException(404, "movie not set")
    if t is None:
        # 原样返回完整文件（不支持拖动，但可用于小文件或兜底）
        return FileResponse(p, media_type="video/mp4", filename=Path(p).name)
    # 从 t 秒起播（推荐）
    return stream_mp4_from_time(p, t)

@app.get("/video/clip")
def video_clip(t: float | None = Query(None)):
    p = STATE.paths.get("clip")
    if not p or not Path(p).exists():
        raise HTTPException(404, "clip not set")
    if t is None:
        return FileResponse(p, media_type="video/mp4", filename=Path(p).name)
    return stream_mp4_from_time(p, t)

def stream_mp4_from_time(path: str, t: float):
    """
    用 ffmpeg 无重编码从 t 秒开始输出可流式 MP4（frag mp4）。
    要求系统有 ffmpeg 命令。
    """
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")

    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, t)),
        "-i", path,
        "-c", "copy",
        "-movflags", "+faststart+frag_keyframe+empty_moov",
        "-f", "mp4", "-",
    ]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg not found; please install ffmpeg and ensure it's in PATH")
    if not proc.stdout:
        raise HTTPException(500, "ffmpeg no stdout")

    def _cleanup():
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    headers = {
        "Cache-Control": "no-store",
        "Accept-Ranges": "bytes",
    }
    return StreamingResponse(proc.stdout, media_type="video/mp4", headers=headers, background=BackgroundTask(_cleanup))

@app.get("/scenes")
def scenes():
    return {"ok": True, "scenes": _scenes_summary()}

@app.get("/segments")
def segments(clip_scene_id: int = Query(...)):
    groups = group_by_clip_scene(STATE.match_segments)
    segs = groups.get(clip_scene_id, [])
    out=[]
    for r in segs:
        seg_id = r.get("seg_id")
        mo = STATE.applied_changes.get(seg_id, r.get("matched_orig_seg"))
        out.append({
            "seg_id": seg_id,
            "clip": {"scene_seg_idx": r.get("scene_seg_idx"), "start": r.get("start"), "end": r.get("end"), "scene_id": r.get("scene_id")},
            "matched_orig_seg": mo,
            "top_matches": r.get("top_matches", []),
        })
    return out

@app.post("/apply")
def apply(req: ApplyBatchReq):
    for ch in req.changes:
        if isinstance(ch.seg_id, int):
            STATE.applied_changes[ch.seg_id] = ch.chosen
    sidecar = _save_overrides_from_state()
    return {"ok": True, "applied": len(req.changes), "sidecar": str(sidecar)}

@app.get("/overrides")
def get_overrides():
    return {"ok": True, "path": str(_overrides_path()), "count": len(STATE.applied_changes or {}), "data": STATE.applied_changes}


@app.post("/overrides/clear")
def clear_overrides():
    STATE.applied_changes = {}
    p = _overrides_path()
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass
    return {"ok": True}

@app.post("/save")
def save(req: SaveReq):
    arr = []
    for r in STATE.match_segments:
        cp = dict(r)
        seg_id = r.get("seg_id")
        if seg_id in STATE.applied_changes:
            cp["matched_orig_seg"] = STATE.applied_changes[seg_id]
        arr.append(cp)
    out_path = req.out_path or str(Path(STATE.project_root or ".") / "match_segments_exported.json")
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    return {"ok": True, "path": out_path}
