from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import logging
import math
import copy
from ..rerank.score_fusion import ScoreFusion, FusionParams
from ..priors.local_state import LocalState
def _round_ms(x: float, ms: int = 3) -> int:
    # return integer milliseconds (ms=3) or decimilliseconds if ms=2, etc.
    if x is None:
        return -1
    return int(round(float(x) * (10 ** ms)))

def load_movie_meta_segs(segs_path: Path | None, store_root: Path | None = None):
    """Read movie meta segs and build lookup tables.
    If segs_path is provided, use it; else if store_root is provided, use store_root/movie/meta/segs.json.
    Returns (exact_map, by_scene).
    """
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

    def _round_ms(x: float, ms: int = 3) -> int:
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
        key = (_round_ms(t0), _round_ms(t1), scene_id)
        exact_map[key] = rec
        by_scene.setdefault(scene_id, []).append(rec)

    for sid in list(by_scene.keys()):
        by_scene[sid].sort(key=lambda r: r.get("start", 0.0))
    return exact_map, by_scene


def attach_seg_ids_from_meta(item: dict, exact_map: dict, by_scene: dict, tol_sec: float = 0.03) -> dict:
    """Fill seg_id / scene_seg_idx for a movie candidate item using movie meta segs.
    If either field is missing/None, try to backfill from (start,end,scene_id) match.
    Returns the mutated item.
    """
    need_seg_id = item.get("seg_id") is None
    need_scene_seg_idx = item.get("scene_seg_idx") is None
    if not (need_seg_id or need_scene_seg_idx):
        return item
    s = float(item.get("start", 0.0))
    e = float(item.get("end", 0.0))
    sid = item.get("scene_id")
    # exact match by rounded milliseconds
    key = (_round_ms(s), _round_ms(e), sid)
    rec = exact_map.get(key)
    if rec is None:
        # fuzzy within tol
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
        # Only overwrite missing (None) fields
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
def _read_clip_frames(path: Path, n: int = 24) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.zeros((0,), dtype=np.float32)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(0, frame_count-1), num=max(1, n), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    cap.release()
    if not frames:
        return np.zeros((0,), dtype=np.float32)
    # stack to [T,H,W,3]
    return np.stack(frames, axis=0)

from ..matcher.query_io import load_queries
from ..matcher.stage1_retriever import Stage1Retriever
from ..matcher.stage2_reranker import Stage2Reranker

from ..utils.io import write_json
from ..utils.faiss_utils import IdMap
from ..types import CropVariant
import csv

# Helper: lightweight explanation writer (can move to recmatcher/utils/explain_writer.py later)
class ExplainWriter:
    """
    Lightweight JSONL writer for per-segment explanations.
    It emits:
      - a single 'meta' line with run-level information
      - one 'segment' line per clip segment
    """
    def __init__(self, path: Path, level: str = "full"):
        self.path = Path(path)
        self.level = level
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # open in text mode, utf-8, overwrite on each run
        self._fh = open(self.path, "w", encoding="utf-8")
    
    def _write(self, obj: dict):
        import json as _json
        self._fh.write(_json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
    
    def write_meta(self, meta: dict):
        rec = {"type": "meta"}
        rec.update(meta or {})
        self._write(rec)
    
    def write_segment(self, seg: dict):
        rec = {"type": "segment"}
        rec.update(seg or {})
        self._write(rec)
    
    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

#
# tag -> 索引名 的默认映射（按我们之前的方案）
VARIANT_MAP = {
    "tight": "centercrop",
    "context": "letterbox",
    "left": "h_left",
    "right": "h_right",
    "center": "h_center",  # 若电影端建了 center
}

# 变体权重与融合温度（轻量校准+投票融合用）
VAR_WEIGHTS = {
    "centercrop": 1.00,
    "letterbox": 0.98,
    "h_left": 0.97,
    "h_center": 0.97,
    "h_right": 0.97,
}
VOTE_TAU = 0.06  # legacy default (kept for meta only)
MIRROR_PENALTY = 0.97  # 镜像命中小幅降权

def _compute_variant_calib(retriever, seg_bags, available_variants, sample_per_var: int = 128):
    """估计每个变体的一阶段分数尺度（中位数与IQR），用于跨变体校准。
    返回 {variant(str): (median, iqr)}，若无样本则给出稳健默认值。
    """
    import numpy as _np
    calib = {}
    by_var_vecs = {v: [] for v in available_variants}
    # 从 seg_bags 里抽取每个变体的若干查询向量（只取前若干个，足够估计尺度）
    for _sid, bag in seg_bags.items():
        for var, vec_items in bag.items():
            if var not in by_var_vecs:
                continue
            for it in vec_items:
                by_var_vecs[var].append(it["vec"])  # it 是 {"vec": ..., "mirrored": bool}
                if len(by_var_vecs[var]) >= sample_per_var:
                    break
    for var, arr in by_var_vecs.items():
        if not arr:
            calib[var] = (0.0, 0.06)  # 缺省尺度
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
    """
    在 store_root 下同时搜：
      - movie/index/<variant>.faiss
      - movie/emb/<variant>_id_map.csv
    二者都存在才纳入。
    额外返回 `id_rows`: {variant(str): List[dict]}，便于按行号取元数据。
    """
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
            # 原有 IdMap（若别处需要）
            id_maps[var] = IdMap.from_csv(idmap_path)
            # 读原始行，后续我们按行号拿时间/scene等
            rows = []
            with open(idmap_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Debug columns once per variant
                try:
                    print(f"[idmap] {var.value} columns:", reader.fieldnames)
                except Exception:
                    pass

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
                    seg_id = _pick_num(row, [
                        "seg_id", "segment_id", "orig_seg_id", "seg_idx", "idx", "id"
                    ], cast=int, default=None)

                    scene_seg_idx = _pick_num(row, [
                        "scene_seg_idx", "seg_idx", "scene_seg_index"
                    ], cast=int, default=None)

                    start_v = _pick_float(row, [
                        "start", "t0", "ts", "begin", "s"
                    ], default=0.0)

                    end_v = _pick_float(row, [
                        "end", "t1", "te", "finish", "e"
                    ], default=0.0)

                    scene_id = _pick_num(row, [
                        "scene_id", "scene", "scene_idx", "sceneindex"
                    ], cast=int, default=None)

                    faiss_id = _pick_num(row, ["faiss_id", "fid", "index"], cast=int, default=None)
                    movie_id = row.get("movie_id", None)
                    shot_id = _pick_num(row, ["shot_id", "shot", "shot_idx"], cast=int, default=None)

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

    print("[indices] found:", {k.value if hasattr(k, "value") else k for k in index_paths.keys()})
    return index_paths, id_maps, id_rows

def _temporal_nms(items, win_sec: float = 3.0):
    """Greedy temporal NMS on already-materialized dict items with 'start','end','score'.
    Keeps highest-score item per overlapping window of +/- win_sec around 'start'.
    """
    if not items:
        return items
    items = sorted(items, key=lambda x: x.get("start", 0.0))
    kept = []
    i = 0
    while i < len(items):
        s0 = items[i]["start"]
        # collect a window
        bucket = []
        j = i
        while j < len(items) and abs(items[j]["start"] - s0) <= win_sec:
            bucket.append(items[j])
            j += 1
        # pick max score in bucket
        best = max(bucket, key=lambda x: x.get("score", 0.0))
        kept.append(best)
        i = j
    return kept

def main():
    ap = argparse.ArgumentParser("recmatcher-match-from-queries")
    ap.add_argument("--queries", required=True, help="encode_clip 产出目录（包含 queries/ 和 meta/）")
    ap.add_argument("--clip_segs", required=True, help="同 encode 阶段的分段 JSON（用于输出对齐）")
    ap.add_argument("--store", required=True, help="电影索引根目录（含 movie/index/*.faiss）")
    ap.add_argument("--out", required=True, help="输出 match.json")
    ap.add_argument("--explain_out", required=False, help="解释输出 JSONL（默认与 --out 同目录的 match_explain.jsonl）")
    ap.add_argument("--explain_level", choices=["basic", "full"], default="full", help="解释详细程度：basic=关键字段；full=完整字段")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--movie_segs", required=False, help="电影原片分段 JSON（segs.json）路径；用于按 (t0,t1,scene_id) 反查 seg_id/scene_seg_idx")
    ap.add_argument("--skip_movie", action="store_true", help="二阶段重排时跳过取电影原片帧")
    ap.add_argument("--clip", required=False, help="原短片路径（用于二阶段参考帧），推荐提供")
    ap.add_argument("--no_norm", action="store_true", help="禁用对查询向量的L2归一化（默认会归一化）")
    ap.add_argument("--nms_sec", type=float, default=0.0, help="对输出做简单时间NMS的窗口秒数(>0生效)")
    ap.add_argument("--debug_probe", type=int, default=0, help="调试：只取前N条查询做探针并打印每个variant的top1得分(0=关闭)")
    ap.add_argument("--debug_dump_norms", action="store_true", help="调试：打印查询向量的范数统计")
    ap.add_argument("--skip_rerank", action="store_true", help="完全跳过二阶段重排（仅输出一阶段得分与候选）")
    ap.add_argument("--score_source", choices=["vp", "rerank", "max", "both"], default="vp",
                    help="选择输出的score来源：vp=一阶段相似度；rerank=二阶段分；max=两者取最大；both=同时写入两个分并以vp作为score")
    # Fusion / priors parameters (all optional; defaults keep behavior close to old)
    ap.add_argument("--tau", type=float, default=0.8, help="soft-vote temperature (higher=flatter)")
    ap.add_argument("--z_clip", type=float, default=5.0, help="clip standardized z to [-z_clip, z_clip]")
    ap.add_argument("--w_consensus", type=float, default=0.05, help="bonus per additional participating variant")
    ap.add_argument("--w_len", type=float, default=0.2, help="duration consistency penalty weight")
    ap.add_argument("--cont_w", type=float, default=0.12, help="contiguity (siblings/same-scene) weight")
    ap.add_argument("--mode_w", type=float, default=0.2, help="mode prior weight")
    ap.add_argument("--mode_window", type=int, default=10, help="sliding window for mode prior")
    # Uncertainty export (optional)
    ap.add_argument("--uncert_ratio_thr", type=float, default=2.0, help="top1/top2 vote ratio threshold for uncertainty")
    ap.add_argument("--len_dev_thr", type=float, default=0.25, help="abs(duration_ratio-1) threshold for uncertainty")
    ap.add_argument("--consensus_min", type=int, default=2, help="minimum participating variants to avoid 'single_view' uncertainty")
    ap.add_argument("--uncert_out", required=False, help="optional path to dump uncertain seg list")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.clip_segs, "r", encoding="utf-8") as f:
        clip_segs = json.load(f)
    # 期望 clip_segs 是 list，每项至少有 seg_id/start/end/scene_id/scene_seg_idx

    qdir = Path(args.queries)
    qlist = load_queries(qdir)
    
    index_paths, id_maps, id_rows = load_indices(Path(args.store))
    segs_path = Path(args.movie_segs) if getattr(args, "movie_segs", None) else None
    movie_exact, movie_by_scene = load_movie_meta_segs(segs_path, Path(args.store))
    retr = Stage1Retriever(index_paths=index_paths, id_maps=id_maps, topk=int(args.topk))

    # 取得可用索引的变体名集合（统一成字符串，如 'letterbox'）
    if hasattr(retr, "_indices"):
        keys = list(retr._indices.keys())
    elif hasattr(retr, "indices"):
        keys = list(retr.indices.keys())
    else:
        keys = []
    available_variants = { (k.value if hasattr(k, "value") else str(k)) for k in keys }
    print("[stage1] available variants:", sorted(available_variants))

    # 把 tag 映射成 retriever 能识别的 variant（和 store 的索引名一致），并把 vecs[N,D] 展开为逐条 vec[D]
    qlist_mapped = []  # flat list: [{'variant': ..., 'tag': ..., 'mirrored': ..., 'vec': np.ndarray[D]}, ...]
    for q in qlist:
        tag = q["tag"]
        var = VARIANT_MAP.get(tag)
        if var is None or var not in available_variants:
            continue
        vecs = q["vecs"]
        if vecs is None:
            continue
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if vecs.shape[0] == 0:
            continue
        for i in range(vecs.shape[0]):
            v = vecs[i].astype(np.float32, copy=False)
            if not args.no_norm:
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-6:
                    v = v / nrm
            qlist_mapped.append({
                "variant": var,
                "tag": tag,
                "mirrored": bool(q.get("mirrored", False)),
                "vec": v,
            })

    if args.debug_dump_norms:
        norms = []
        for q in qlist_mapped:
            v = q["vec"]
            norms.append(float(np.linalg.norm(v)))
        if norms:
            norms = np.array(norms, dtype=np.float32)
            print(f"[debug] query norms: min={norms.min():.4f} med={np.median(norms):.4f} max={norms.max():.4f}")
        else:
            print("[debug] query norms: EMPTY")

    if not qlist_mapped:
        raise SystemExit("No valid query vectors after mapping; check your --queries directory and VARIANT_MAP/indices alignment.")

    # 将扁平的查询向量按 seg_id 聚合：{ seg_id: { variant: [vec, ...] } }
    # 假设每个 tag 的 vecs 按 clip_segs 顺序排列
    seg_bags = {}
    num_segs = len(clip_segs) if isinstance(clip_segs, list) else 0
    # 先初始化 seg 容器
    for s in clip_segs:
        seg_bags[s.get("seg_id")] = {}
    # 重新从原始 qlist（带 vecs 批量）做一次聚合，避免丢失段位次序
    for q in qdir.iterdir() if False else qlist:  # 占位防静态检查；真实用 qlist
        tag = q["tag"]
        var = VARIANT_MAP.get(tag)
        if var is None:
            continue
        vecs = q["vecs"]
        if vecs is None:
            continue
        # 保障 2D 形状 [N,D]
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        n = vecs.shape[0]
        # 与 clip_segs 对齐，取 min 防止越界
        L = min(n, num_segs)
        for i in range(L):
            sid = clip_segs[i].get("seg_id")
            v = vecs[i].astype(np.float32, copy=False)
            if not args.no_norm:
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-6:
                    v = v / nrm
            seg_bags[sid].setdefault(var, []).append({"vec": v, "mirrored": bool(q.get("mirrored", False))})

    def _probe_variants(retriever, qitems, topk=5):
        """Return dict: {variant: top1_list} for the first few queries per variant."""
        by_var = {}
        for qi in qitems:
            by_var.setdefault(qi["variant"], []).append(qi)
        report = {}
        for var, items in by_var.items():
            D, I, idm = retriever._search_variant(var, np.stack([x["vec"] for x in items]).astype(np.float32))
            # D shape [N, topk]
            top1 = D[:, 0].tolist() if D.size else []
            report[var] = top1
        return report

    if args.debug_probe and args.debug_probe > 0:
        N = int(args.debug_probe)
        sample = qlist_mapped[:N]
        rep = _probe_variants(retr, sample, topk=min(args.topk, 10))
        print("[probe] per-variant top1 stats on first", N, "queries:")
        for var, arr in rep.items():
            if arr:
                a = np.array(arr, dtype=np.float32)
                print(f"  - {var:11s}: n={len(arr):4d} min={a.min():.3f} med={np.median(a):.3f} max={a.max():.3f}")
            else:
                print(f"  - {var:11s}: n=0")

    def _row_start_end(m):
        # m 已经标准化了 start/end，但为了保险再兜底一下
        s = m.get("start")
        e = m.get("end")
        if s is None:
            s = m.get("t0", 0.0)
        if e is None:
            e = m.get("t1", 0.0)
        return float(s or 0.0), float(e or 0.0)

    # 基于当前查询包估计各变体分数尺度，用于跨变体校准
    calib_tbl = _compute_variant_calib(retr, seg_bags, available_variants)

    # Initialize explain writer (JSONL)
    explain_path = Path(args.explain_out) if getattr(args, "explain_out", None) else Path(args.out).with_name("match_explain.jsonl")
    exp_writer = ExplainWriter(explain_path, level=getattr(args, "explain_level", "full"))

    # Initialize fusion and local priors
    fusion = ScoreFusion(FusionParams(
        tau=float(args.tau),
        z_clip=float(args.z_clip),
        var_weights=VAR_WEIGHTS,
        mirror_penalty=MIRROR_PENALTY,
        w_consensus=float(args.w_consensus),
        w_len=float(args.w_len),
        topk=int(args.topk),
    ))
    priors = LocalState(window=int(args.mode_window), w_cont=float(args.cont_w), w_mode=float(args.mode_w))

    # Emit run-level meta once
    run_meta = {
        "available_variants": sorted(list(available_variants)),
        "variant_calibration": {k: {"median": float(v[0]), "iqr": float(v[1])} for k, v in (calib_tbl or {}).items()},
        "weights": VAR_WEIGHTS,
        "vote_tau": float(args.tau),
        "z_clip": float(args.z_clip),
        "mirror_penalty": MIRROR_PENALTY,
        "w_consensus": float(args.w_consensus),
        "w_len": float(args.w_len),
        "cont_w": float(args.cont_w),
        "mode_w": float(args.mode_w),
        "mode_window": int(args.mode_window),
        "topk": int(args.topk),
        "nms_sec": float(args.nms_sec or 0.0),
        "score_source": args.score_source,
        "movie": args.movie,
        "uncertainty": {
            "ratio_thr": float(args.uncert_ratio_thr),
            "len_dev_thr": float(args.len_dev_thr),
            "consensus_min": int(args.consensus_min),
        },
    }
    exp_writer.write_meta(run_meta)

    # Thin wrapper kept for compatibility; real fusion is below in-loop
    def retrieve_one_seg(seg_id, bag, topk):
        return []

    # === Per-seg 检索与融合，输出分段结果 ===
    results = []
    for s in clip_segs:
        sid = s.get("seg_id")
        bag = seg_bags.get(sid, {})
        # Fuse per-seg candidates with new fusion logic
        clip_dur = max(1e-9, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
        if not bag:
            top_items_pre = []
        else:
            top_items_pre = fusion.fuse_segment(
                seg_id=sid,
                bag=bag,
                retriever=retr,
                id_rows=id_rows,
                calib_tbl=calib_tbl,
                clip_dur=clip_dur,
                topk=int(args.topk),
            )
        # 标注 NMS 前排名
        for _i, _it in enumerate(top_items_pre):
            _it["pre_nms_rank"] = int(_i + 1)

        # 应用局部先验（仅加分，不做硬约束）
        prev_sel = getattr(priors, "prev_selected", None)
        for _it in top_items_pre:
            # contiguity
            b_cont, cont_detail = priors.contiguity_bonus(prev_sel, _it)
            if b_cont > 0:
                _it["vote"] = float(_it.get("vote", 0.0)) + float(b_cont)
                _it.setdefault("reason_codes", []).append("cont+")
            _it["contiguity_feat"] = cont_detail
            # mode prior
            pv = _it.get("primary_variant")
            b_mode, mode_detail = priors.mode_prior_bonus(pv)
            if b_mode > 0:
                _it["vote"] = float(_it.get("vote", 0.0)) + float(b_mode)
                _it.setdefault("reason_codes", []).append("mode+")
            _it["mode_prior"] = mode_detail

        # 备份一份用于 explain（深拷贝）
        explain_pre = [copy.deepcopy(x) for x in top_items_pre[:max(10, int(args.topk))]]

        # 可选 NMS
        if args.nms_sec and args.nms_sec > 0:
            top_items = _temporal_nms(list(top_items_pre), win_sec=float(args.nms_sec))
        else:
            top_items = list(top_items_pre)

        # 最终排序
        top_items.sort(key=lambda x: (-float(x.get("vote", x.get("score", 0.0))),
                                      -float(x.get("score", 0.0)),
                                      float(x.get("start", 0.0)),
                                      float(x.get("end", 0.0))))
        for _i, _it in enumerate(top_items):
            _it["post_nms_rank"] = int(_i + 1)

        # Wrap matched_orig_seg and top_matches properly
        def wrap_items(items):
            wrapped = []
            for item in items:
                wrapped.append({
                    "seg_id": item.get("seg_id"),
                    "scene_seg_idx": item.get("scene_seg_idx"),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "scene_id": item.get("scene_id"),
                    "score": item.get("score"),
                    "faiss_id": item.get("faiss_id"),
                    "movie_id": item.get("movie_id"),
                    "shot_id": item.get("shot_id"),
                })
            return wrapped
        wrapped_top_items = wrap_items(top_items)[:10]
        matched_orig_seg = wrapped_top_items[0] if wrapped_top_items else None

        # Uncertainty measures & update priors' state
        uncert = {}
        if len(top_items) >= 1:
            top1 = top_items[0]
            top2 = top_items[1] if len(top_items) > 1 else None
            ratio = float(top1.get("vote", 0.0)) / (float(top2.get("vote", 0.0)) + 1e-9) if top2 else float("inf")
            consensus_m = int(len(top1.get("source_variants") or []))
            len_dev = float(abs(float(top1.get("duration_ratio", 1.0)) - 1.0))
            flags = []
            if ratio < float(args.uncert_ratio_thr):
                flags.append("low_ratio")
            if consensus_m < int(args.consensus_min):
                flags.append("single_view")
            if len_dev > float(args.len_dev_thr):
                flags.append("len_dev")
            uncert = {"ratio": ratio, "consensus_m": consensus_m, "len_dev": len_dev, "flags": flags}
            # update priors with selected (pseudo anchor)
            priors.prev_selected = top1
            try:
                priors.update_mode_hist(top1)
            except Exception:
                pass

        # 解释输出（每段一行）
        try:
            seg_explain = {
                "seg_id": sid,
                "clip": {
                    "scene_seg_idx": s.get("scene_seg_idx"),
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "scene_id": s.get("scene_id"),
                },
                "candidates_pre": explain_pre[:10],
                "candidates_post": [copy.deepcopy(x) for x in top_items[:10]],
                "selected_index": 0 if top_items else None,
                "uncertainty": uncert,
            }
            exp_writer.write_segment(seg_explain)
        except Exception:
            # 解释输出失败不影响主流程
            pass

        out = {
            "seg_id": sid,
            "scene_seg_idx": s.get("scene_seg_idx"),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "scene_id": s.get("scene_id"),
            "matched_orig_seg": matched_orig_seg,
            "top_matches": wrapped_top_items,
        }
        results.append(out)

    # 统计一下整体分数分布（基于 matched_orig_seg）
    scores = [x["matched_orig_seg"]["score"] for x in results if x.get("matched_orig_seg")]
    if scores:
        import numpy as _np
        scores = _np.array(scores, dtype=_np.float32)
        print(f"[summary] matched top1: min={scores.min():.3f} med={_np.median(scores):.3f} max={scores.max():.3f} (n={len(scores)})")

    # Optionally dump uncertain segment ids (placeholder; UI can also read from JSONL)
    if getattr(args, "uncert_out", None):
        try:
            with open(args.uncert_out, "w", encoding="utf-8") as f:
                json.dump({"segments": []}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out), results)
    try:
        exp_writer.close()
    except Exception:
        pass
    return


if __name__ == "__main__":
    main()
