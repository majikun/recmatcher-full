from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import logging
import copy
from ..rerank.score_fusion import ScoreFusion, FusionParams
from ..priors.local_state import LocalState
from ..post.scene_consensus import SceneConsensus, SceneConsensusParams
from ..post.scene_chain import SceneChain, SceneChainParams, GlobalTimeChain, GlobalChainParams

def _round_ms(x: float, ms: int = 3) -> int:
    if x is None:
        return -1
    return int(round(float(x) * (10 ** ms)))

def load_movie_meta_segs(segs_path: Path | None, store_root: Path | None = None):
    if segs_path is not None:
        meta_path = Path(segs_path)
    elif store_root is not None:
        meta_path = store_root / "movie" / "meta" / "segs.json"
    else:
        meta_path = None

    exact_map = {}
    by_scene = {}
    if meta_path is None or not meta_path.exists():
        return exact_map, by_scene

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return exact_map, by_scene

    def _pick(d, keys, default=None, cast=None):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                try:
                    return cast(d[k]) if cast else d[k]
                except Exception:
                    continue
        return default

    def _round_ms_local(x: float, ms: int = 3) -> int:
        if x is None:
            return -1
        return int(round(float(x) * (10 ** ms)))

    for row in (data or []):
        seg_id = _pick(row, ["seg_id", "id", "seg_idx", "segment_id"], default=None, cast=int)
        scene_seg_idx = _pick(row, ["scene_seg_idx", "seg_idx", "scene_seg_index"], default=None, cast=int)
        t0 = _pick(row, ["start", "t0", "ts", "begin"], default=0.0, cast=float)
        t1 = _pick(row, ["end", "t1", "te", "finish"], default=0.0, cast=float)
        scene_id = _pick(row, ["scene_id", "scene", "scene_idx"], default=None, cast=int)
        rec = {
            "seg_id": seg_id,
            "scene_seg_idx": scene_seg_idx,
            "start": float(t0 or 0.0),
            "end": float(t1 or 0.0),
            "scene_id": scene_id,
        }
        key = (_round_ms_local(t0), _round_ms_local(t1), scene_id)
        exact_map[key] = rec
        by_scene.setdefault(scene_id, []).append(rec)

    for sid in list(by_scene.keys()):
        by_scene[sid].sort(key=lambda r: r.get("start", 0.0))
    return exact_map, by_scene

def attach_seg_ids_from_meta(item: dict, exact_map: dict, by_scene: dict, tol_sec: float = 0.03) -> dict:
    need_seg_id = item.get("seg_id") is None
    need_scene_seg_idx = item.get("scene_seg_idx") is None
    if not (need_seg_id or need_scene_seg_idx):
        return item
    s = float(item.get("start", 0.0))
    e = float(item.get("end", 0.0))
    sid = item.get("scene_id")
    key = (_round_ms(s), _round_ms(e), sid)
    rec = exact_map.get(key)
    if rec is None:
        tol = float(tol_sec)
        bucket = by_scene.get(sid, [])
        best = None
        best_err = 1e9
        for r in bucket:
            err = max(abs(r["start"] - s), abs(r["end"] - e))
            if err <= tol and err < best_err:
                best = r
                best_err = err
        rec = best
    if rec:
        if need_seg_id and rec.get("seg_id") is not None:
            try:
                item["seg_id"] = int(rec.get("seg_id"))
            except Exception:
                item["seg_id"] = rec.get("seg_id")
        if need_scene_seg_idx and rec.get("scene_seg_idx") is not None:
            try:
                item["scene_seg_idx"] = int(rec.get("scene_seg_idx"))
            except Exception:
                item["scene_seg_idx"] = rec.get("scene_seg_idx")
    return item

from ..matcher.query_io import load_queries
from ..matcher.stage1_retriever import Stage1Retriever
from ..matcher.stage2_reranker import Stage2Reranker
from ..utils.io import write_json
from ..utils.faiss_utils import IdMap
from ..types import CropVariant
import csv

class ExplainWriter:
    def __init__(self, path: Path, level: str = "full"):
        self.path = Path(path)
        self.level = level
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
    def _write(self, obj: dict):
        import json as _json
        self._fh.write(_json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
    def write_meta(self, meta: dict):
        rec = {"type": "meta"}
        rec.update(meta or {})
        self._write(rec)
    def write_any(self, payload: dict):
        # payload should include a 'type' key, e.g., 'scene_meta', 'scene_chain', etc.
        self._write(payload or {})
    def write_segment(self, seg: dict):
        rec = {"type": "segment"}
        rec.update(seg or {})
        self._write(rec)
    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

VARIANT_MAP = {
    "tight": "centercrop",
    "context": "letterbox",
    "left": "h_left",
    "right": "h_right",
    "center": "h_center",
}

VAR_WEIGHTS = {
    "centercrop": 1.00,
    "letterbox": 0.98,
    "h_left": 0.97,
    "h_center": 0.97,
    "h_right": 0.97,
}
MIRROR_PENALTY = 0.97

def _compute_variant_calib(retriever, seg_bags, available_variants, sample_per_var: int = 128):
    import numpy as _np
    calib = {}
    by_var_vecs = {v: [] for v in available_variants}
    for _sid, bag in seg_bags.items():
        for var, vec_items in bag.items():
            if var not in by_var_vecs:
                continue
            for it in vec_items:
                by_var_vecs[var].append(it["vec"])
                if len(by_var_vecs[var]) >= sample_per_var:
                    break
    for var, arr in by_var_vecs.items():
        if not arr:
            calib[var] = (0.0, 0.06)
            continue
        X = _np.stack(arr).astype(_np.float32)
        try:
            D, I, _idm = retriever._search_variant(var, X)
            if D.size == 0:
                calib[var] = (0.0, 0.06)
                continue
            top1 = D[:, 0]
            med = float(_np.median(top1))
            q25, q75 = _np.percentile(top1, [25, 75])
            iqr = float(max(q75 - q25, 1e-3))
            calib[var] = (med, iqr)
        except Exception:
            calib[var] = (0.0, 0.06)
    return calib

def load_indices(store_root: Path):
    index_dir = store_root / "movie" / "index"
    idmap_dir = store_root / "movie" / "emb"

    index_paths = {}
    id_maps = {}
    id_rows = {}

    variants = [
        CropVariant.LETTERBOX,
        CropVariant.CENTERCROP,
        CropVariant.H_LEFT,
        CropVariant.H_CENTER,
        CropVariant.H_RIGHT,
    ]

    for var in variants:
        faiss_path = index_dir / f"{var.value}.faiss"
        idmap_path = idmap_dir / f"{var.value}_id_map.csv"
        if faiss_path.exists() and idmap_path.exists():
            index_paths[var] = str(faiss_path)
            id_maps[var] = IdMap.from_csv(idmap_path)
            rows = []
            with open(idmap_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                def _pick_num(d, keys, cast=float, default=None):
                    for k in keys:
                        if k in d and d[k] not in (None, ""):
                            try:
                                return cast(d[k])
                            except Exception:
                                continue
                    return default
                def _pick_float(d, keys, default=0.0):
                    return _pick_num(d, keys, cast=float, default=default)
                for row in reader:
                    seg_id = _pick_num(row, ["seg_id","segment_id","orig_seg_id","seg_idx","idx","id"], cast=int, default=None)
                    scene_seg_idx = _pick_num(row, ["scene_seg_idx","seg_idx","scene_seg_index"], cast=int, default=None)
                    start_v = _pick_float(row, ["start","t0","ts","begin","s"], default=0.0)
                    end_v = _pick_float(row, ["end","t1","te","finish","e"], default=0.0)
                    scene_id = _pick_num(row, ["scene_id","scene","scene_idx","sceneindex"], cast=int, default=None)
                    faiss_id = _pick_num(row, ["faiss_id","fid","index"], cast=int, default=None)
                    movie_id = row.get("movie_id", None)
                    shot_id = _pick_num(row, ["shot_id","shot","shot_idx"], cast=int, default=None)
                    rows.append({
                        "faiss_id": faiss_id,
                        "movie_id": movie_id,
                        "shot_id": shot_id,
                        "seg_id": seg_id,
                        "scene_seg_idx": scene_seg_idx,
                        "start": start_v,
                        "end": end_v,
                        "scene_id": scene_id,
                    })
            id_rows[var.value] = rows
    return index_paths, id_maps, id_rows

def _temporal_nms(items, win_sec: float = 3.0):
    if not items:
        return items
    items = sorted(items, key=lambda x: x.get("start", 0.0))
    kept = []
    i = 0
    while i < len(items):
        s0 = items[i]["start"]
        bucket = []
        j = i
        while j < len(items) and abs(items[j]["start"] - s0) <= win_sec:
            bucket.append(items[j]); j += 1
        best = max(bucket, key=lambda x: x.get("score", 0.0))
        kept.append(best)
        i = j
    return kept

def main():
    ap = argparse.ArgumentParser("recmatcher-match-from-queries")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--clip_segs", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--explain_out", required=False)
    ap.add_argument("--explain_level", choices=["basic","full"], default="full")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--movie_segs", required=False)
    ap.add_argument("--skip_movie", action="store_true")
    ap.add_argument("--clip", required=False)
    ap.add_argument("--no_norm", action="store_true")
    ap.add_argument("--nms_sec", type=float, default=0.0)
    ap.add_argument("--debug_probe", type=int, default=0)
    ap.add_argument("--debug_dump_norms", action="store_true")
    ap.add_argument("--skip_rerank", action="store_true")
    ap.add_argument("--score_source", choices=["vp","rerank","max","both"], default="vp")
    # Fusion / priors
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--z_clip", type=float, default=5.0)
    ap.add_argument("--w_consensus", type=float, default=0.05)
    ap.add_argument("--w_len", type=float, default=0.2)
    ap.add_argument("--cont_w", type=float, default=0.12)
    ap.add_argument("--mode_w", type=float, default=0.2)
    ap.add_argument("--mode_window", type=int, default=10)
    # Uncertainty
    ap.add_argument("--uncert_ratio_thr", type=float, default=2.0)
    ap.add_argument("--len_dev_thr", type=float, default=0.25)
    ap.add_argument("--consensus_min", type=int, default=2)
    ap.add_argument("--uncert_out", required=False)
    # Scene consensus / chain (new)
    ap.add_argument("--scene_enable", action="store_true", help="enable scene-level consensus + chain post-processing")
    ap.add_argument("--scene_vote_top_n", type=int, default=3)
    ap.add_argument("--scene_dynamic_k", action="store_true")
    ap.add_argument("--scene_continuity_eps", type=float, default=0.08)
    ap.add_argument("--scene_continuity_max_gap", type=int, default=2)
    ap.add_argument("--scene_chain_max_skip", type=int, default=1)
    ap.add_argument("--scene_chain_len_w", type=float, default=0.3)
    ap.add_argument("--scene_chain_jump_penalty", type=float, default=0.2)
    ap.add_argument("--fill_enable", action="store_true")
    ap.add_argument("--fill_max_gap", type=int, default=2)
    ap.add_argument("--fill_penalty", type=float, default=0.05)
    ap.add_argument("--scene_out", required=False, help="optional path to dump scene-level mapping/chain summary")
    # Global cross-clip-scene time chain (optional)
    ap.add_argument("--global_chain_enable", action="store_true", help="enable cross-clip-scene global time-chain DP")
    ap.add_argument("--global_topk", type=int, default=3, help="candidates per clip scene for global chain")
    ap.add_argument("--global_alpha", type=float, default=0.6, help="penalty for time going backwards (flashback)")
    ap.add_argument("--global_beta", type=float, default=0.2, help="penalty for large forward jumps")
    ap.add_argument("--global_gamma", type=float, default=0.2, help="penalty for movie-scene switches between adjacent clip scenes")
    ap.add_argument("--global_allow_flashbacks", type=int, default=1, help="how many flashbacks are allowed globally")
    ap.add_argument("--global_score_norm", choices=["minmax","softmax","none"], default="minmax", help="per-step normalization for consensus scores")
    ap.add_argument("--global_score_w", type=float, default=1.0, help="weight for (normalized) score term in global chain")
    ap.add_argument("--global_softmax_tau", type=float, default=0.5, help="temperature for softmax normalization (if selected)")
    ap.add_argument("--global_stick_bonus", type=float, default=0.15, help="reward for staying in the same movie scene (reduces switchiness)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.clip_segs, "r", encoding="utf-8") as f:
        clip_segs = json.load(f)

    qdir = Path(args.queries)
    qlist = load_queries(qdir)
    index_paths, id_maps, id_rows = load_indices(Path(args.store))
    segs_path = Path(args.movie_segs) if getattr(args, "movie_segs", None) else None
    movie_exact, movie_by_scene = load_movie_meta_segs(segs_path, Path(args.store))
    retr = Stage1Retriever(index_paths=index_paths, id_maps=id_maps, topk=int(args.topk))

    # available variants
    if hasattr(retr, "_indices"):
        keys = list(retr._indices.keys())
    elif hasattr(retr, "indices"):
        keys = list(retr.indices.keys())
    else:
        keys = []
    available_variants = { (k.value if hasattr(k, "value") else str(k)) for k in keys }

    # map tag->variant and normalize vectors
    qlist_mapped = []
    for q in qlist:
        tag = q["tag"]
        var = {"tight":"centercrop","context":"letterbox","left":"h_left","right":"h_right","center":"h_center"}.get(tag)
        if var is None or var not in available_variants:
            continue
        vecs = q["vecs"]
        if vecs is None:
            continue
        if vecs.ndim == 1: vecs = vecs[None, :]
        if vecs.shape[0] == 0:
            continue
        for i in range(vecs.shape[0]):
            v = vecs[i].astype(np.float32, copy=False)
            if not args.no_norm:
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-6: v = v / nrm
            qlist_mapped.append({"variant": var, "tag": tag, "mirrored": bool(q.get("mirrored", False)), "vec": v})

    if not qlist_mapped:
        raise SystemExit("No valid query vectors after mapping; check queries & indices.")

    # seg_bags: seg_id -> {variant: [ {vec, mirrored}, ... ]}
    seg_bags = {}; num_segs = len(clip_segs) if isinstance(clip_segs, list) else 0
    for s in clip_segs:
        seg_bags[s.get("seg_id")] = {}
    for q in qdir.iterdir() if False else qlist:
        tag = q["tag"]; var = {"tight":"centercrop","context":"letterbox","left":"h_left","right":"h_right","center":"h_center"}.get(tag)
        if var is None: continue
        vecs = q["vecs"]; 
        if vecs is None: continue
        if vecs.ndim == 1: vecs = vecs[None, :]
        L = min(vecs.shape[0], num_segs)
        for i in range(L):
            sid = clip_segs[i].get("seg_id")
            v = vecs[i].astype(np.float32, copy=False)
            if not args.no_norm:
                nrm = float(np.linalg.norm(v)); v = v / nrm if nrm > 1e-6 else v
            seg_bags[sid].setdefault(var, []).append({"vec": v, "mirrored": bool(q.get("mirrored", False))})

    # variant calibration
    calib_tbl = _compute_variant_calib(retr, seg_bags, available_variants)

    explain_path = Path(args.explain_out) if getattr(args, "explain_out", None) else Path(args.out).with_name("match_explain.jsonl")
    exp_writer = ExplainWriter(explain_path, level=getattr(args, "explain_level", "full"))

    fusion = ScoreFusion(FusionParams(
        tau=float(args.tau), z_clip=float(args.z_clip),
        var_weights={"centercrop":1.0,"letterbox":0.98,"h_left":0.97,"h_center":0.97,"h_right":0.97},
        mirror_penalty=0.97, w_consensus=float(args.w_consensus), w_len=float(args.w_len), topk=int(args.topk)))
    priors = LocalState(window=int(args.mode_window), w_cont=float(args.cont_w), w_mode=float(args.mode_w))

    run_meta = {
        "available_variants": sorted(list(available_variants)),
        "variant_calibration": {k: {"median": float(v[0]), "iqr": float(v[1])} for k, v in (calib_tbl or {}).items()},
        "weights": {"centercrop":1.0,"letterbox":0.98,"h_left":0.97,"h_center":0.97,"h_right":0.97},
        "vote_tau": float(args.tau),
        "z_clip": float(args.z_clip),
        "mirror_penalty": 0.97,
        "w_consensus": float(args.w_consensus),
        "w_len": float(args.w_len),
        "cont_w": float(args.cont_w),
        "mode_w": float(args.mode_w),
        "mode_window": int(args.mode_window),
        "topk": int(args.topk),
        "nms_sec": float(args.nms_sec or 0.0),
        "score_source": args.score_source,
        "movie": args.movie,
        "uncertainty": {"ratio_thr": float(args.uncert_ratio_thr), "len_dev_thr": float(args.len_dev_thr), "consensus_min": int(args.consensus_min)},
        "scene": {
            "enable": bool(args.scene_enable),
            "vote_top_n": int(args.scene_vote_top_n),
            "dynamic_k": bool(args.scene_dynamic_k),
            "continuity_eps": float(args.scene_continuity_eps),
            "continuity_max_gap": int(args.scene_continuity_max_gap),
            "chain_max_skip": int(args.scene_chain_max_skip),
            "chain_len_w": float(args.scene_chain_len_w),
            "chain_jump_penalty": float(args.scene_chain_jump_penalty),
            "fill_enable": bool(args.fill_enable),
            "fill_max_gap": int(args.fill_max_gap),
            "fill_penalty": float(args.fill_penalty),
        },
        "global_chain": {
            "enable": bool(args.global_chain_enable),
            "topk": int(args.global_topk),
            "alpha": float(args.global_alpha),
            "beta": float(args.global_beta),
            "gamma": float(args.global_gamma),
            "allow_flashbacks": int(args.global_allow_flashbacks),
            "score_norm": str(args.global_score_norm),
            "score_w": float(args.global_score_w),
            "softmax_tau": float(args.global_softmax_tau),
            "stick_bonus": float(args.global_stick_bonus),
        }
    }
    exp_writer.write_meta(run_meta)

    results = []
    per_seg_cands = {}   # seg_id -> ranked candidates (after priors & nms)
    per_seg_uncert = {}  # seg_id -> uncertainty dict
    clip_scene_groups = {}  # clip_scene_id -> list of seg_ids (in clip order)

    # per-seg retrieval + fusion
    for s in clip_segs:
        sid = s.get("seg_id")
        clip_scene_id = s.get("scene_id")
        clip_scene_groups.setdefault(clip_scene_id, []).append(sid)
        bag = seg_bags.get(sid, {})
        clip_dur = max(1e-9, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
        if not bag: top_items_pre = []
        else:
            top_items_pre = fusion.fuse_segment(seg_id=sid, bag=bag, retriever=retr, id_rows=id_rows, calib_tbl=calib_tbl, clip_dur=clip_dur, topk=int(args.topk))
        for _i, _it in enumerate(top_items_pre): _it["pre_nms_rank"] = int(_i + 1)
        prev_sel = getattr(priors, "prev_selected", None)
        for _it in top_items_pre:
            b_cont, cont_detail = priors.contiguity_bonus(prev_sel, _it)
            if b_cont > 0: _it["vote"] = float(_it.get("vote", 0.0)) + float(b_cont); _it.setdefault("reason_codes", []).append("cont+")
            _it["contiguity_feat"] = cont_detail
            pv = _it.get("primary_variant")
            b_mode, mode_detail = priors.mode_prior_bonus(pv)
            if b_mode > 0: _it["vote"] = float(_it.get("vote", 0.0)) + float(b_mode); _it.setdefault("reason_codes", []).append("mode+")
            _it["mode_prior"] = mode_detail
        explain_pre = [copy.deepcopy(x) for x in top_items_pre[:max(10, int(args.topk))]]
        if args.nms_sec and args.nms_sec > 0: top_items = _temporal_nms(list(top_items_pre), win_sec=float(args.nms_sec))
        else: top_items = list(top_items_pre)
        top_items.sort(key=lambda x: (-float(x.get("vote", x.get("score", 0.0))), -float(x.get("score", 0.0)), float(x.get("start", 0.0)), float(x.get("end", 0.0))))
        for _i, _it in enumerate(top_items): _it["post_nms_rank"] = int(_i + 1)
        try:
            for _it in top_items: attach_seg_ids_from_meta(_it, movie_exact, movie_by_scene)
            for _it in explain_pre: attach_seg_ids_from_meta(_it, movie_exact, movie_by_scene)
        except Exception: pass

        # uncertainty
        uncert = {}; 
        if len(top_items) >= 1:
            top1 = top_items[0]; top2 = top_items[1] if len(top_items) > 1 else None
            ratio = float(top1.get("vote", 0.0)) / (float(top2.get("vote", 0.0)) + 1e-9) if top2 else float("inf")
            consensus_m = int(len(top1.get("source_variants") or []))
            len_dev = float(abs(float(top1.get("duration_ratio", 1.0)) - 1.0))
            flags = []; 
            if ratio < float(args.uncert_ratio_thr): flags.append("low_ratio")
            if consensus_m < int(args.consensus_min): flags.append("single_view")
            if len_dev > float(args.len_dev_thr): flags.append("len_dev")
            uncert = {"ratio": ratio, "consensus_m": consensus_m, "len_dev": len_dev, "flags": flags}
            priors.prev_selected = top1
            try: priors.update_mode_hist(top1)
            except Exception: pass
        per_seg_uncert[sid] = uncert
        per_seg_cands[sid] = top_items

        # explain per segment now (pre scene)
        try:
            seg_explain = {
                "type": "segment",
                "seg_id": sid,
                "clip": {"scene_seg_idx": s.get("scene_seg_idx"), "start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "scene_id": s.get("scene_id")},
                "candidates_pre": explain_pre[:10],
                "candidates_post": [copy.deepcopy(x) for x in top_items[:10]],
                "selected_index": 0 if top_items else None,
                "uncertainty": uncert,
            }
            exp_writer.write_any(seg_explain)
        except Exception:
            pass

        # default result using top1 (before scene post-processing)
        def wrap_items(items):
            return [{
                "seg_id": it.get("seg_id"),
                "scene_seg_idx": it.get("scene_seg_idx"),
                "start": it.get("start"),
                "end": it.get("end"),
                "scene_id": it.get("scene_id"),
                "score": it.get("score"),
                "faiss_id": it.get("faiss_id"),
                "movie_id": it.get("movie_id"),
                "shot_id": it.get("shot_id"),
            } for it in items]
        wrapped_top_items = wrap_items(top_items)[:10]
        matched_orig_seg = wrapped_top_items[0] if wrapped_top_items else None
        results.append({
            "seg_id": sid,
            "scene_seg_idx": s.get("scene_seg_idx"),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "scene_id": s.get("scene_id"),
            "matched_orig_seg": matched_orig_seg,
            "top_matches": wrapped_top_items,
        })

    # -------- Scene-level post processing --------
    scene_summary = {}
    if args.scene_enable:
        sc_params = SceneConsensusParams(
            vote_top_n=int(args.scene_vote_top_n),
            dynamic_k=bool(args.scene_dynamic_k),
            continuity_eps=float(args.scene_continuity_eps),
            continuity_max_gap=int(args.scene_continuity_max_gap),
        )
        ch_params = SceneChainParams(
            max_skip=int(args.scene_chain_max_skip),
            len_w=float(args.scene_chain_len_w),
            jump_penalty=float(args.scene_chain_jump_penalty),
            fill_enable=bool(args.fill_enable),
            fill_max_gap=int(args.fill_max_gap),
            fill_penalty=float(args.fill_penalty),
        )
        sc_engine = SceneConsensus(sc_params)
        ch_engine = SceneChain(ch_params)

        # helper: representative time (median of starts within this clip-scene & movie-scene)
        def _rep_time_for_scene(seg_ids, target_scene_id):
            times = []
            for sid in seg_ids:
                for c in per_seg_cands.get(sid, []):
                    if c.get("scene_id") == target_scene_id and c.get("start") is not None:
                        try:
                            times.append(float(c.get("start", 0.0)))
                        except Exception:
                            pass
            if times:
                times.sort()
                mid = len(times) // 2
                return times[mid] if len(times) % 2 == 1 else 0.5 * (times[mid-1] + times[mid])
            seq = movie_by_scene.get(target_scene_id, [])
            if seq:
                return float(seq[0].get("start", 0.0))
            return 0.0

        scene_candidates_series = {}   # clip_scene_id -> [{'scene_id','score','rep_time',...}, ...]
        selected_scene_map = {}        # clip_scene_id -> selected scene by consensus (pre-global)

        # build quick index of result entries by seg_id
        res_by_sid = {r["seg_id"]: r for r in results}
        # group objects for consensus: list of dict per seg {seg_id, candidates, uncertainty}
        for clip_scene_id, seg_ids in clip_scene_groups.items():
            group_objs = []
            for sid in seg_ids:
                group_objs.append({
                    "seg_id": sid,
                    "candidates": per_seg_cands.get(sid, []),
                    "uncertainty": per_seg_uncert.get(sid, {}),
                })
            consensus = sc_engine.aggregate(group_objs, movie_by_scene)
            # explain scene meta
            try:
                exp_writer.write_any({
                    "type": "scene_meta",
                    "clip_scene_id": clip_scene_id,
                    "scores": consensus.get("scores", {}),
                    "selected_scene": consensus.get("selected_scene"),
                    "coverage": consensus.get("coverage", {}),
                    "avg_len_dev": consensus.get("avg_len_dev", {}),
                    "tie_break": consensus.get("tie_break", ""),
                })
            except Exception:
                pass
            # collect candidates for global chain
            selected_scene_map[clip_scene_id] = consensus.get("selected_scene")
            cand_list = []
            for msid, sc in (consensus.get("scores") or {}).items():
                if msid is None:
                    continue
                rep_t = _rep_time_for_scene(seg_ids, msid)
                cand_list.append({
                    "scene_id": msid,
                    "score": float(sc),
                    "rep_time": float(rep_t),
                    "coverage": int((consensus.get("coverage") or {}).get(msid, 0)),
                    "avg_len_dev": float((consensus.get("avg_len_dev") or {}).get(msid, 0.0)),
                })
            scene_candidates_series[clip_scene_id] = cand_list
            selected_scene = consensus.get("selected_scene")
            if selected_scene is None:
                continue

            # chain & optional fills
            chain = ch_engine.build(group_objs, selected_scene, movie_by_scene)
            assign = chain.get("assign", {})
            fills = chain.get("fills", []); stats = chain.get("stats", {})
            # apply assignment to results (update matched_orig_seg & top_matches filtered)
            for sid in seg_ids:
                res = res_by_sid.get(sid)
                if res is None:
                    continue
                chosen = assign.get(sid)
                # always filter top_matches to the selected movie scene to keep UI focused
                filt = [c for c in per_seg_cands.get(sid, []) if c.get("scene_id") == selected_scene]
                res["top_matches"] = [{
                    "seg_id": c.get("seg_id"),
                    "scene_seg_idx": c.get("scene_seg_idx"),
                    "start": c.get("start"),
                    "end": c.get("end"),
                    "scene_id": c.get("scene_id"),
                    "score": c.get("score"),
                    "faiss_id": c.get("faiss_id"),
                    "movie_id": c.get("movie_id"),
                    "shot_id": c.get("shot_id"),
                } for c in filt[:10]]

                if chosen is not None:
                    attach_seg_ids_from_meta(chosen, movie_exact, movie_by_scene)
                    res["matched_orig_seg"] = {
                        "seg_id": chosen.get("seg_id"),
                        "scene_seg_idx": chosen.get("scene_seg_idx"),
                        "start": chosen.get("start"),
                        "end": chosen.get("end"),
                        "scene_id": chosen.get("scene_id"),
                        "score": chosen.get("score"),
                        "faiss_id": chosen.get("faiss_id"),
                        "movie_id": chosen.get("movie_id"),
                        "shot_id": chosen.get("shot_id"),
                    }
                else:
                    # strict chain leaves a hole at this seg; clear stale selection to avoid duplicates
                    res["matched_orig_seg"] = None
            # explain chain
            try:
                exp_writer.write_any({
                    "type": "scene_chain",
                    "clip_scene_id": clip_scene_id,
                    "movie_scene_id": selected_scene,
                    "assign": {int(k): (v.get("scene_seg_idx") if v else None) for k,v in assign.items()},
                    "fills": fills,
                    "stats": stats,
                })
            except Exception:
                pass

            scene_summary[int(clip_scene_id) if clip_scene_id is not None else -1] = {
                "movie_scene_id": selected_scene,
                "assign": {int(k): (int(v.get("scene_seg_idx")) if v and v.get("scene_seg_idx") is not None else None) for k,v in assign.items()},
                "fills": fills,
                "stats": stats,
            }

        # ---- Global time-chain over clip scenes (optional) ----
        if args.global_chain_enable:
            # Build clip-scene order from traversal of clip_segs (deduplicate consecutive)
            clip_scene_order = []
            for s in clip_segs:
                csid = s.get("scene_id")
                if not clip_scene_order or clip_scene_order[-1] != csid:
                    clip_scene_order.append(csid)

            # Build series for DP from collected candidates
            series = []
            for csid in clip_scene_order:
                cands = scene_candidates_series.get(csid, [])
                cands = sorted(cands, key=lambda x: -x.get("score", 0.0))[:max(1, int(args.global_topk))]
                series.append(cands)

            gc_params = GlobalChainParams(
                alpha=float(args.global_alpha),
                beta=float(args.global_beta),
                gamma=float(args.global_gamma),
                allow_flashbacks=int(args.global_allow_flashbacks),
                topk=int(args.global_topk),
                score_w=float(args.global_score_w),
                score_norm=str(args.global_score_norm),
                softmax_tau=float(args.global_softmax_tau),
                stick_bonus=float(args.global_stick_bonus),
            )
            gchain = GlobalTimeChain(gc_params)
            gplan = gchain.plan(series)

            # explain: global chain summary
            try:
                exp_writer.write_any({
                    "type": "global_chain",
                    "clip_scene_order": clip_scene_order,
                    "choice_scene": gplan.get("choice_scene", []),
                    "flashbacks": gplan.get("flashbacks", []),
                    "cost": gplan.get("cost", 0.0),
                })
            except Exception:
                pass

            # Apply overrides where global plan disagrees with local consensus
            override_map = {clip_scene_order[i]: gplan["choice_scene"][i] if i < len(gplan.get("choice_scene", [])) else None for i in range(len(clip_scene_order))}
            res_by_sid = {r["seg_id"]: r for r in results}  # rebuild index
            for csid in clip_scene_order:
                chosen_scene = override_map.get(csid)
                if chosen_scene is None or chosen_scene == selected_scene_map.get(csid):
                    continue  # no change
                seg_ids = clip_scene_groups.get(csid, [])
                group_objs = []
                for sid in seg_ids:
                    group_objs.append({
                        "seg_id": sid,
                        "candidates": per_seg_cands.get(sid, []),
                        "uncertainty": per_seg_uncert.get(sid, {}),
                    })
                chain = ch_engine.build(group_objs, chosen_scene, movie_by_scene)
                assign = chain.get("assign", {}); fills = chain.get("fills", []); stats = chain.get("stats", {})
                for sid in seg_ids:
                    res = res_by_sid.get(sid)
                    if res is None:
                        continue
                    chosen = assign.get(sid)
                    # always filter to the globally chosen movie scene
                    filt = [c for c in per_seg_cands.get(sid, []) if c.get("scene_id") == chosen_scene]
                    res["top_matches"] = [{
                        "seg_id": c.get("seg_id"),
                        "scene_seg_idx": c.get("scene_seg_idx"),
                        "start": c.get("start"),
                        "end": c.get("end"),
                        "scene_id": c.get("scene_id"),
                        "score": c.get("score"),
                        "faiss_id": c.get("faiss_id"),
                        "movie_id": c.get("movie_id"),
                        "shot_id": c.get("shot_id"),
                    } for c in filt[:10]]

                    if chosen is not None:
                        attach_seg_ids_from_meta(chosen, movie_exact, movie_by_scene)
                        res["matched_orig_seg"] = {
                            "seg_id": chosen.get("seg_id"),
                            "scene_seg_idx": chosen.get("scene_seg_idx"),
                            "start": chosen.get("start"),
                            "end": chosen.get("end"),
                            "scene_id": chosen.get("scene_id"),
                            "score": chosen.get("score"),
                            "faiss_id": chosen.get("faiss_id"),
                            "movie_id": chosen.get("movie_id"),
                            "shot_id": chosen.get("shot_id"),
                        }
                    else:
                        # clear stale result if global chain leaves a gap under strict rule
                        res["matched_orig_seg"] = None
                # optional: update scene_summary entry if present
                if csid in scene_summary:
                    scene_summary[csid]["movie_scene_id"] = chosen_scene
                    scene_summary[csid]["stats"] = stats

    # write outputs
    if getattr(args, "uncert_out", None):
        try:
            with open(args.uncert_out, "w", encoding="utf-8") as f:
                json.dump({"segments": []}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out), results)
    if args.scene_enable and getattr(args, "scene_out", None):
        try:
            with open(args.scene_out, "w", encoding="utf-8") as f:
                json.dump(scene_summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    try:
        exp_writer.close()
    except Exception:
        pass
    return

if __name__ == "__main__":
    main()
