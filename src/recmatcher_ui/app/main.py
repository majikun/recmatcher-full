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
