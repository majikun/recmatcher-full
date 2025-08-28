from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .state import STATE
from .schemas import OpenProjectReq, ApplyBatchReq, SaveReq
from .utils import group_by_clip_scene
from pathlib import Path
import json

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
    scenes = _scenes_summary()
    return {"ok": True, "scenes": scenes, "paths": STATE.paths}

@app.get("/video/movie")
def video_movie():
    p = STATE.paths.get("movie")
    if not p or not Path(p).exists(): raise HTTPException(404,"movie not set")
    return FileResponse(p, media_type="video/mp4", filename=Path(p).name)

@app.get("/video/clip")
def video_clip():
    p = STATE.paths.get("clip")
    if not p or not Path(p).exists(): raise HTTPException(404,"clip not set")
    return FileResponse(p, media_type="video/mp4", filename=Path(p).name)

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
    return {"ok": True, "applied": len(req.changes)}

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
