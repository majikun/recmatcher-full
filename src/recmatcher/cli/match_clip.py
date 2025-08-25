from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import cv2
from ..utils.logging import setup_logging
from ..utils.io import load_yaml, read_json, write_json
from ..types import CropVariant, Candidate, TimeWindow
from ..utils.faiss_utils import FaissIndex, IdMap
from ..matcher.query_readout import QueryReadout
from ..matcher.stage1_retriever import Stage1Retriever
from ..matcher.stage2_reranker import Stage2Reranker
from ..matcher.movie_frame_provider import FFMPEGFrameProvider
from ..matcher.scene_aggregator import SceneAggregator
from ..matcher.sequence_aligner import SequenceAligner

def _scan_store(root: Path):
    """Return dict: movie_id -> variant -> paths (index, id_map), and movie file path if known."""
    store = {}
    for movie_dir in root.iterdir():
        if not movie_dir.is_dir(): continue
        mid = movie_dir.name
        emb = movie_dir / "emb"; idxd = movie_dir / "index"
        print(f"Scanning movie '{root}/{mid}' ...")
        if not emb.exists() or not idxd.exists(): continue
        var2 = {}
        for var in ("letterbox","centercrop","h_left","h_center","h_right"):
            ip = idxd / f"{var}.faiss"; mp = emb / f"{var}_id_map.csv"
            if ip.exists() and mp.exists():
                var2[var] = {"index": str(ip), "id_map": str(mp)}
        if var2:
            store[mid] = var2
    print(f"Found {len(store)} movies with indices under store.")
    return store

def _read_clip_frames(clip_path: str, t0: float, t1: float, n: int=9):
    print(f"Reading clip frames from {clip_path} [{t0:.2f},{t1:.2f}] ...")
    cap = cv2.VideoCapture(str(clip_path))
    ts = np.linspace(t0, t1, max(2,n))
    frames = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000.0)
        ok, f = cap.read()
        if not ok: continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    cap.release()
    return frames

def main():
    ap = argparse.ArgumentParser(description="Match clip segments to movie.")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--clip_segs", required=True, help="clip segs json (list of {start,end,scene_id})")
    ap.add_argument("--store", required=True, help="root dir that contains movie emb/index")
    ap.add_argument("--movie", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="override default config yaml")
    ap.add_argument("--skip_movie", action="store_true", help="skip loading original movie frames during rerank")
    ap.add_argument("--debug", action="store_true", help="enable debug logging")
    args = ap.parse_args()

    log = setup_logging("DEBUG" if args.debug else "INFO")
    cfg = load_yaml(args.config) if args.config else load_yaml(Path(__file__).resolve().parents[2] / "recmatcher/config" / "default.yaml")
    segs = read_json(args.clip_segs)

    # build index mapping
    store = _scan_store(Path(args.store))
    if not store:
        raise SystemExit("No movie indices found under store.")

    # Build variant -> (FaissIndex, IdMap) across all movies (concat later if desired).
    # For simplicity we search per movie and merge.
    all_results = {"segments": []}
    qr = QueryReadout(cfg["query_readout"], size=cfg["movie_index"]["size_base"], n_frames=cfg["query_readout"]["n_frames"], agg=cfg["query_readout"]["frames_aggregate"])
    rerank = Stage2Reranker(cfg["rerank"])
    mreader = FFMPEGFrameProvider(movie_path=Path(args.movie))
    aggregator = SceneAggregator()
    aligner = SequenceAligner()

    # For each segment, search across each movie and keep best candidates
    from ..utils.faiss_utils import IdMap, FaissIndex
    seg2cands = {}
    for si, seg in enumerate(segs):
        t0=float(seg["start"]); t1=float(seg["end"]); scene_id=int(seg.get("scene_id",-1))
        clip_frames = _read_clip_frames(args.clip, t0, t1, n=cfg["query_readout"]["n_frames"]*3)
        qlist = qr.make_queries(clip_frames)
        merged_cands = []
        for mid, var2 in store.items():
            index_paths = {}
            id_maps = {}
            for var, paths in var2.items():
                cv = CropVariant(var)
                index_paths[cv] = paths["index"]
                id_maps[cv] = IdMap(paths["id_map"])
            retr = Stage1Retriever(index_paths, id_maps, topk=cfg["retrieval"]["topk"])
            cands = retr.search(qlist)
            # attach movie file hint
            for c in cands:
                c.tw = TimeWindow(movie_id=mid, t0=c.tw.t0, t1=c.tw.t1, scene_id=c.tw.scene_id, shot_id=c.tw.shot_id)
            merged_cands.extend(cands)
        # Re-rank using H-Tile/dynamic
        if args.skip_movie:
            clip_arr = np.zeros((0,))
            merged_cands = rerank.score(clip_arr, merged_cands, movie_reader=None)
        else:
            clip_arr = np.stack([f.astype(np.float32)/255.0 for f in clip_frames], axis=0) if clip_frames else np.zeros((0,))
            merged_cands = rerank.score(clip_arr, merged_cands, movie_reader=mreader)
        seg2cands[si] = merged_cands

    # Simple align (monotonic across segments)
    aligned = aligner.align_monotonic({}, seg2cands)
    for si, cand in aligned.items():
        all_results["segments"].append({
            "clip_seg_id": int(si),
            "movie_id": cand.tw.movie_id,
            "t0": cand.tw.t0,
            "t1": cand.tw.t1,
            "variant": cand.variant.value,
            "fused": cand.score_fused,
            "scores": {"vp": cand.score_vp, "tile": cand.score_tile, "dynamic": cand.score_dyn, "cut": cand.score_cut, "prior": cand.score_prior},
            "source": {"index": cand.source_index, "faiss_id": cand.source_id},
            "explain": cand.explain or {}
        })

    write_json(args.out, all_results)
    log.info(f"Wrote results to {args.out}")
