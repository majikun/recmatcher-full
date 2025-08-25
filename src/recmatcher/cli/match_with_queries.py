from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
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
    """
    index_dir = store_root / "movie" / "index"
    idmap_dir = store_root / "movie" / "emb"

    index_paths = {}
    id_maps = {}

    # 你的 CropVariant 应该定义了这些 value：letterbox / centercrop / h_left / h_center / h_right
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
            
    print("[indices] found:", {k.value if hasattr(k, "value") else k for k in index_paths.keys()})
    return index_paths, id_maps

def main():
    ap = argparse.ArgumentParser("recmatcher-match-from-queries")
    ap.add_argument("--queries", required=True, help="encode_clip 产出目录（包含 queries/ 和 meta/）")
    ap.add_argument("--clip_segs", required=True, help="同 encode 阶段的分段 JSON（用于输出对齐）")
    ap.add_argument("--store", required=True, help="电影索引根目录（含 movie/index/*.faiss）")
    ap.add_argument("--out", required=True, help="输出 match.json")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--skip_movie", action="store_true", help="二阶段重排时跳过取电影原片帧")
    ap.add_argument("--clip", required=False, help="原短片路径（用于二阶段参考帧），推荐提供")
    args = ap.parse_args()

    qdir = Path(args.queries)
    qlist = load_queries(qdir)
    
    index_paths, id_maps = load_indices(Path(args.store))
    retr = Stage1Retriever(index_paths=index_paths, id_maps=id_maps, topk=int(args.topk))

    # 把 tag 映射成 retriever 能识别的 variant（和 store 的索引名一致），并把 vecs[N,D] 展开为逐条 vec[D]
    qlist_mapped = []  # flat list: [{'variant': ..., 'tag': ..., 'mirrored': ..., 'vec': np.ndarray[D]}, ...]
    for q in qlist:
        tag = q["tag"]
        var = VARIANT_MAP.get(tag)
        if var is None or var not in retr._indices:
            # 没有对应索引就跳过该视角
            continue
        vecs = q["vecs"]
        if vecs is None:
            continue
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if vecs.shape[0] == 0:
            continue
        for i in range(vecs.shape[0]):
            qlist_mapped.append({
                "variant": var,
                "tag": tag,
                "mirrored": bool(q.get("mirrored", False)),
                "vec": vecs[i].astype(np.float32, copy=False),
            })

    if not qlist_mapped:
        raise SystemExit("No valid query vectors after mapping; check your --queries directory and VARIANT_MAP/indices alignment.")

    # 一阶段召回
    cands = retr.search(qlist_mapped)

    # 二阶段重排（可选电影帧 + 可选短片参考帧）
    rerank = Stage2Reranker(config={})
    if args.skip_movie:
        clip_arr = np.zeros((0,), dtype=np.float32)
        cands = rerank.score(clip_arr, cands, movie_reader=None)
    else:
        from ..matcher.movie_frame_provider import FFMPEGFrameProvider
        mreader = FFMPEGFrameProvider(movie_path=Path(args.movie))
        # 提供短片参考帧：如果没传 --clip，则给空（旧版会在内部做降级）；建议传入以避免 NaN
        if args.clip:
            clip_arr = _read_clip_frames(Path(args.clip), n=24)
        else:
            clip_arr = np.zeros((0,), dtype=np.float32)
        cands = rerank.score(clip_arr, cands, movie_reader=mreader)

    # 输出 match.json（尽量兼容已有 to_dict；否则写基础字段）
    out_items = []
    for c in cands:
        if hasattr(c, "to_dict"):
            out_items.append(c.to_dict())
        else:
            t0 = getattr(c, "t0", getattr(getattr(c, "tw", c), "t0", 0.0))
            t1 = getattr(c, "t1", getattr(getattr(c, "tw", c), "t1", 0.0))
            out_items.append({
                "seg_id": getattr(c, "seg_id", None),
                "scene_seg_idx": getattr(c, "scene_seg_idx", None),
                "start": float(t0),
                "end": float(t1),
                "scene_id": getattr(c, "scene_id", None),
                "score": float(getattr(c, "score_fused", getattr(c, "score_vp", 0.0))),
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out), out_items)


if __name__ == "__main__":
    main()