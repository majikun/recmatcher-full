from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import logging
import math
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

# tag -> 索引名 的默认映射（按我们之前的方案）
VARIANT_MAP = {
    "tight": "centercrop",
    "context": "letterbox",
    "left": "h_left",
    "right": "h_right",
    "center": "h_center",  # 若电影端建了 center
}

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
            seg_bags[sid].setdefault(var, []).append(v)

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

    def retrieve_one_seg(seg_id, bag, topk):
        """bag: {variant(str): [vec,...]} → 融合多个变体，返回排序后的候选 list[dict]。"""
        fuse = {}  # key=(start,end,scene_id) → score(max)
        meta = {}  # 同 key → 任意一条元信息（含 seg_id/scene_seg_idx）
        for var, vec_list in bag.items():
            if var not in available_variants:
                continue
            X = np.stack(vec_list).astype(np.float32)
            D, I, _idm = retr._search_variant(var, X)
            rows = id_rows.get(var, [])
            for r in range(I.shape[0]):
                for k in range(min(I.shape[1], topk)):
                    j = int(I[r, k])
                    if j < 0 or j >= len(rows):
                        continue
                    m = rows[j]
                    ss, ee = _row_start_end(m)
                    key = (ss, ee, m.get("scene_id"))
                    sc = float(D[r, k])
                    if key not in fuse or sc > fuse[key]:
                        fuse[key] = sc
                        meta[key] = m
        # 排序并裁剪
        items = []
        for key, sc in fuse.items():
            m = meta[key]
            ss, ee = _row_start_end(m)
            cand = {
                "seg_id": m.get("seg_id"),
                "scene_seg_idx": m.get("scene_seg_idx"),
                "start": ss,
                "end": ee,
                "scene_id": m.get("scene_id"),
                "score": sc,
                "faiss_id": m.get("faiss_id"),
                "movie_id": m.get("movie_id"),
                "shot_id": m.get("shot_id"),
            }
            cand = attach_seg_ids_from_meta(cand, movie_exact, movie_by_scene)
            items.append(cand)
        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:topk]

    # === Per-seg 检索与融合，输出分段结果 ===
    results = []
    for s in clip_segs:
        sid = s.get("seg_id")
        bag = seg_bags.get(sid, {})
        if not bag:
            top_items = []
        else:
            top_items = retrieve_one_seg(sid, bag, topk=int(args.topk))
            if args.nms_sec and args.nms_sec > 0:
                top_items = _temporal_nms(top_items, win_sec=float(args.nms_sec))
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
        scores = np.array(scores, dtype=np.float32)
        print(f"[summary] matched top1: min={scores.min():.3f} med={np.median(scores):.3f} max={scores.max():.3f} (n={len(scores)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out), results)
    return


if __name__ == "__main__":
    main()