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
from typing import List, Tuple

# --- keyframes helpers --------------------------------------------------------

def _kf_path(kind: str) -> Path:
    root = Path(STATE.project_root or ".")
    name = f"{kind}_keyframes.json"
    return root / name


def _ffprobe_keyframes(input_path: str) -> List[float]:
    """Return list of keyframe timestamps (seconds) for the first video stream.
    Tries several ffprobe strategies for robustness:
      1) frames + -skip_frame nokey (lightweight) using best_effort_timestamp_time
      2) frames (all) filter key_frame==1 or pict_type==I
      3) packets filter flags contains 'K'
    """
    def _unique_sorted(vals: List[float]) -> List[float]:
        out = sorted(set(float(x) for x in vals if x is not None))
        # ensure 0.0 exists
        if not out or out[0] > 0.0:
            out = [0.0] + out
        return out

    # Strategy 1: only keyframes via decoder skip, JSON frames
    try:
        cmd1 = [
            "ffprobe", "-v", "error",
            "-skip_frame", "nokey",
            "-select_streams", "v:0",
            "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time",
            "-of", "json",
            input_path,
        ]
        out1 = subprocess.check_output(cmd1, stderr=subprocess.STDOUT)
        data1 = json.loads(out1.decode("utf-8", errors="ignore"))
        vals1: List[float] = []
        for fr in (data1.get("frames") or []):
            ts = fr.get("best_effort_timestamp_time") or fr.get("pkt_pts_time")
            if ts is None:
                continue
            try:
                vals1.append(float(ts))
            except Exception:
                pass
        if len(vals1) > 1:
            return _unique_sorted(vals1)
    except Exception:
        pass

    # Strategy 2: all frames, filter key_frame==1 or pict_type==I
    try:
        cmd2 = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_frames",
            "-show_entries", "frame=key_frame,pict_type,best_effort_timestamp_time,pkt_pts_time",
            "-of", "json",
            input_path,
        ]
        out2 = subprocess.check_output(cmd2, stderr=subprocess.STDOUT)
        data2 = json.loads(out2.decode("utf-8", errors="ignore"))
        vals2: List[float] = []
        for fr in (data2.get("frames") or []):
            try:
                is_k = int(fr.get("key_frame") or 0) == 1
            except Exception:
                is_k = False
            pict = (fr.get("pict_type") or "").upper()
            if is_k or pict == "I":
                ts = fr.get("best_effort_timestamp_time") or fr.get("pkt_pts_time")
                if ts is None:
                    continue
                try:
                    vals2.append(float(ts))
                except Exception:
                    pass
        if len(vals2) > 1:
            return _unique_sorted(vals2)
    except Exception:
        pass

    # Strategy 3: packets, filter flags contains 'K'
    try:
        cmd3 = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_packets",
            "-show_entries", "packet=pts_time,flags",
            "-of", "json",
            input_path,
        ]
        out3 = subprocess.check_output(cmd3, stderr=subprocess.STDOUT)
        data3 = json.loads(out3.decode("utf-8", errors="ignore"))
        vals3: List[float] = []
        for pk in (data3.get("packets") or []):
            flags = str(pk.get("flags") or "")
            if "K" in flags.upper():
                ts = pk.get("pts_time")
                if ts is None:
                    continue
                try:
                    vals3.append(float(ts))
                except Exception:
                    pass
        if len(vals3) > 1:
            return _unique_sorted(vals3)
    except Exception:
        pass

    # Fallback
    return [0.0]


def _save_keyframes(kind: str, src_path: str, kf_list: List[float]) -> Path:
    p = _kf_path(kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "source": src_path,
        "count": len(kf_list),
        "keyframes": kf_list,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def _load_keyframes(kind: str) -> List[float]:
    p = _kf_path(kind)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            arr = obj.get("keyframes") if isinstance(obj, dict) else None
            if isinstance(arr, list):
                return [float(x) for x in arr]
        except Exception:
            pass
    return []


def _ensure_keyframes(kind: str) -> List[float]:
    # in-memory cache on STATE
    if not hasattr(STATE, "keyframes"):
        STATE.keyframes = {}
    cache: dict = getattr(STATE, "keyframes")
    if kind in cache and isinstance(cache[kind], list) and cache[kind]:
        return cache[kind]
    path = (STATE.paths or {}).get(kind)
    if not path or not Path(path).exists():
        cache[kind] = [0.0]
        return cache[kind]
    kf = _load_keyframes(kind)
    if not kf or len(kf) <= 1:
        kf = _ffprobe_keyframes(path)
        _save_keyframes(kind, path, kf)
    cache[kind] = kf
    return kf


def _locate_chunk(kind: str, t: float, d: float, pre: float, post: float) -> Tuple[float, float, float]:
    """Given target start t and duration d, return (base, effective, offset).
    base is the chosen keyframe-aligned start; effective is total chunk duration;
    offset is (t - base) clipped to [0, effective].
    """
    kf = _ensure_keyframes(kind)
    # choose keyframe <= t - pre, else 0.0
    t_pre = max(0.0, (t or 0.0) - max(0.0, pre))
    base = 0.0
    for x in kf:
        if x <= t_pre:
            base = x
        else:
            break
    # ensure we cover till t + d + post
    end_need = max(t, 0.0) + max(d, 0.0) + max(post, 0.0)
    effective = max(0.1, end_need - base)
    offset = max(0.0, min(effective, (t or 0.0) - base))
    return base, effective, offset

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
        applied_map = STATE.applied_changes or {}
        override_count = 0
        try:
            override_count = sum(1 for s in segs if s and s.get("seg_id") in applied_map)
        except Exception:
            override_count = 0
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
            "override_count": override_count,
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
    # Precompute and persist keyframes (movie & clip) for faster, accurate seeks
    try:
        if STATE.paths.get("movie"):
            _ensure_keyframes("movie")
        if STATE.paths.get("clip"):
            _ensure_keyframes("clip")
    except Exception:
        pass
    scenes = _scenes_summary()
    return {"ok": True, "scenes": scenes, "paths": STATE.paths}


# Expose keyframes for debugging or front-end
@app.get("/keyframes")
def keyframes(kind: str = Query("movie"), force: int = Query(0)):
    if force:
        path = (STATE.paths or {}).get(kind)
        if not path or not Path(path).exists():
            raise HTTPException(404, f"{kind} not set")
        kf = _ffprobe_keyframes(path)
        _save_keyframes(kind, path, kf)
        # refresh in-memory cache
        if not hasattr(STATE, "keyframes"):
            STATE.keyframes = {}
        STATE.keyframes[kind] = kf
    else:
        kf = _ensure_keyframes(kind)
    return {"ok": True, "kind": kind, "count": len(kf), "keyframes": kf[:2000]}

@app.get("/video/movie")
def video_movie(t: float | None = Query(None), d: float | None = Query(None), pre: float = Query(0.8), post: float = Query(0.8)):
    p = STATE.paths.get("movie")
    if not p or not Path(p).exists():
        raise HTTPException(404, "movie not set")
    if t is None:
        return FileResponse(p, media_type="video/mp4", filename=Path(p).name)
    # chunked streaming aligned to keyframe
    d_eff = float(d if d is not None else 2.0)
    base, effective, offset = _locate_chunk("movie", float(t), d_eff, pre, post)
    resp = stream_mp4_chunk(p, base, effective)
    # attach locating headers
    try:
        resp.headers["X-Base-Start"] = f"{base:.3f}"
        resp.headers["X-Effective-Duration"] = f"{effective:.3f}"
        resp.headers["X-Offset"] = f"{offset:.3f}"
    except Exception:
        pass
    return resp

@app.get("/video/clip")
def video_clip(t: float | None = Query(None), d: float | None = Query(None), pre: float = Query(0.2), post: float = Query(0.2)):
    p = STATE.paths.get("clip")
    if not p or not Path(p).exists():
        raise HTTPException(404, "clip not set")
    if t is None:
        return FileResponse(p, media_type="video/mp4", filename=Path(p).name)
    d_eff = float(d if d is not None else 2.0)
    base, effective, offset = _locate_chunk("clip", float(t), d_eff, pre, post)
    resp = stream_mp4_chunk(p, base, effective)
    try:
        resp.headers["X-Base-Start"] = f"{base:.3f}"
        resp.headers["X-Effective-Duration"] = f"{effective:.3f}"
        resp.headers["X-Offset"] = f"{offset:.3f}"
    except Exception:
        pass
    return resp

# Endpoint to locate chunk info for a time/duration
@app.get("/video/locate")
def video_locate(kind: str = Query("movie"), t: float = Query(...), d: float = Query(2.0), pre: float = Query(0.8), post: float = Query(0.8)):
    if kind not in ("movie", "clip"):
        raise HTTPException(400, "kind must be 'movie' or 'clip'")
    base, effective, offset = _locate_chunk(kind, t, d, pre, post)
    return {"ok": True, "kind": kind, "t": t, "d": d, "base": base, "effective": effective, "offset": offset}

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

# Streaming helper for chunked aligned streaming
def stream_mp4_chunk(path: str, base: float, duration: float):
    """Stream an fMP4 chunk starting at keyframe-aligned 'base' for 'duration' seconds.
    Uses -c copy with reset timestamps so the fragment timeline starts at 0.
    """
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, base):.3f}",
        "-i", path,
        "-t", f"{max(0.1, duration):.3f}",
        "-reset_timestamps", "1", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart+frag_keyframe+empty_moov",
        "-c", "copy",
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
        "Cache-Control": "public, max-age=3600",
        "Accept-Ranges": "bytes",
        # front-end can read these to compute relative seek within the chunk
        "Access-Control-Expose-Headers": "X-Base-Start, X-Effective-Duration, X-Offset",
    }
    return StreamingResponse(proc.stdout, media_type="video/mp4", headers=headers, background=BackgroundTask(_cleanup))

# --- candidates: expose richer options for a given seg_id ---------------------

def _normalize_cand(x: dict) -> dict:
    """Best-effort normalize candidate dict to expected keys.
    Accepts varied shapes from explain; returns a shallow copy.
    """
    if not isinstance(x, dict):
        return x
    it = dict(x)
    # common aliasing
    if "idx" in it and "scene_seg_idx" not in it:
        it["scene_seg_idx"] = it.get("idx")
    if "seg_idx" in it and "scene_seg_idx" not in it:
        it["scene_seg_idx"] = it.get("seg_idx")
    # sometimes nested under 'orig' / 'movie'
    m = it.get("orig") or it.get("movie") or {}
    for k in ("scene_id", "scene_seg_idx", "start", "end", "faiss_id"):
        if it.get(k) is None and isinstance(m, dict) and m.get(k) is not None:
            it[k] = m.get(k)
    # sometimes score under other name
    if it.get("score") is None:
        for sk in ("rerank", "sim", "s", "prob"):
            if it.get(sk) is not None:
                it["score"] = it.get(sk)
                break
    return it

def _read_explain_slow(seg_id: int):
    """Fallback: linearly scan a jsonl explain file to find the record for seg_id.
    Tries multiple likely locations based on STATE.paths / project_root.
    """
    candidates = []
    try:
        p = None
        # try explicit path from STATE.paths if present
        try:
            p = (getattr(STATE, "paths", {}) or {}).get("explain")
        except Exception:
            p = None
        paths = []
        if p:
            paths.append(Path(p))
        # common fallbacks
        pr = Path(STATE.project_root or ".")
        for name in ("match_explain.jsonl", "match_explain.json", "explain.jsonl"):
            paths.append(pr / name)
        for path in paths:
            try:
                if not path or not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        sid = obj.get("seg_id") or obj.get("clip_seg_id") or obj.get("id")
                        try:
                            if sid is not None and int(sid) == int(seg_id):
                                return obj
                        except Exception:
                            continue
            except Exception:
                continue
    except Exception:
        pass
    return None

def _get_explain_candidates(seg_id: int) -> list[dict]:
    """Read rich candidates from explain jsonl via STATE (if available)."""
    try:
        # STATE.read_explain may raise if offsets not built
        obj = getattr(STATE, "read_explain", None)
        if callable(obj):
            rec = obj(int(seg_id))
        else:
            rec = None  # will try slow scan below
    except Exception:
        rec = None  # will try slow scan below
    if not rec:
        # slow-path scan when offsets are not available
        rec = _read_explain_slow(seg_id)
    items: list[dict] = []
    if isinstance(rec, dict):
        # try common keys (prefer post-processed, then pre-processed)
        for key in ("candidates_post", "candidates_pre", "top_matches", "candidates", "topk", "items"):
            arr = rec.get(key)
            if isinstance(arr, list) and arr:
                items = arr
                break
        # fallback: whole record is a candidate list
        if not items and "scene_id" in rec:
            items = [rec]
    elif isinstance(rec, list):
        items = rec
    # normalize
    out: list[dict] = []
    for it in items:
        if isinstance(it, dict):
            out.append(_normalize_cand(it))
    return out

def _find_seg_row(seg_id: int) -> dict | None:
    for r in STATE.match_segments:
        try:
            if int(r.get("seg_id")) == int(seg_id):
                return r
        except Exception:
            continue
    return None


def _dedup_candidates(items: list[dict]) -> list[dict]:
    """Deduplicate by (scene_id, scene_seg_idx, start, end, faiss_id)."""
    seen = set()
    out: list[dict] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        key = (
            it.get("scene_id"),
            it.get("scene_seg_idx"),
            round(float(it.get("start") or 0.0), 3),
            round(float(it.get("end") or 0.0), 3),
            it.get("faiss_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


@app.get("/candidates")
def candidates(seg_id: int = Query(...), mode: str = Query("top"), k: int = Query(120), offset: int = Query(0)):
    """Return candidates for a clip seg.

    mode:
      - top:   use row.top_matches only
      - scene: filter candidates to same movie scene as the current *applied* (or matched) choice
      - all:   applied (if any) + matched + top_matches, de-duplicated
    New chunked video streaming modes available (see /video/movie, /video/clip endpoints).
    """
    row = _find_seg_row(seg_id)
    if not row:
        raise HTTPException(404, "seg not found")

    applied = (STATE.applied_changes or {}).get(seg_id)
    matched = row.get("matched_orig_seg")
    top = list(row.get("top_matches", [])) or []
    explain_extra = _get_explain_candidates(seg_id)

    def _tag(it: dict, src: str) -> dict:
        if not isinstance(it, dict):
            return it
        it = dict(it)
        it.setdefault("source", src)
        return it

    items: list[dict] = []
    mode = (mode or "top").lower()
    if mode == "top":
        items = [_tag(x, "top") for x in top] + [_tag(x, "explain") for x in explain_extra]
    elif mode == "scene":
        target_scene = None
        if isinstance(applied, dict) and applied.get("scene_id") is not None:
            target_scene = applied.get("scene_id")
        elif isinstance(matched, dict) and matched.get("scene_id") is not None:
            target_scene = matched.get("scene_id")
        base = top
        if target_scene is None:
            items = []
        else:
            items = [_tag(x, "top") for x in base if int(x.get("scene_id") or -1) == int(target_scene)]
            items.extend(_tag(x, "explain") for x in explain_extra if int(x.get("scene_id") or -1) == int(target_scene))
    else:  # all
        if isinstance(applied, dict):
            items.append(_tag(applied, "applied"))
        elif isinstance(matched, dict):
            items.append(_tag(matched, "matched"))
        items.extend(_tag(x, "top") for x in top)
        items.extend(_tag(x, "explain") for x in explain_extra)
    items = _dedup_candidates(items)

    total = len(items)
    sl = items[offset: offset + k]
    return {"ok": True, "seg_id": seg_id, "mode": mode, "total": total, "items": sl}

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
        applied_map = STATE.applied_changes or {}
        is_override = seg_id in applied_map
        mo = applied_map.get(seg_id) or r.get("matched_orig_seg")
        matched_source = "applied" if is_override else "matched"
        out.append({
            "seg_id": seg_id,
            "clip": {
                "scene_seg_idx": r.get("scene_seg_idx"),
                "start": r.get("start"),
                "end": r.get("end"),
                "scene_id": r.get("scene_id"),
            },
            "matched_orig_seg": mo,
            "matched_source": matched_source,
            "is_override": is_override,
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

@app.get("/orig_segments")
def get_orig_segments(scene_id: int = Query(...)):
    """获取指定场景的原始段落列表"""
    try:
        orig_segs_path = Path(STATE.project_root or ".") / "orig_segs_2s.json"
        if not orig_segs_path.exists():
            raise HTTPException(404, "orig_segs_2s.json not found")
        
        with open(orig_segs_path, "r", encoding="utf-8") as f:
            all_segs = json.load(f)
        
        # 过滤出指定场景的段落
        scene_segs = [seg for seg in all_segs if seg.get("scene_id") == scene_id]
        
        return {"ok": True, "scene_id": scene_id, "segments": scene_segs}
    except Exception as e:
        raise HTTPException(500, f"Failed to load orig segments: {str(e)}")


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
