from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from ..matcher.query_readout import QueryReadout
from ..encoder.vpr_embed import VPRemb


def _read_frames_between(cap: cv2.VideoCapture, t0: float, t1: float, n: int) -> List[np.ndarray]:
    """从 [t0,t1] 秒等间隔采 n 帧；返回 RGB uint8 帧列表。"""
    if t1 < t0:
        t0, t1 = t1, t0
    ts = np.linspace(t0, t1, max(2, n))
    frames: List[np.ndarray] = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, f = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


def _save_queries(out_dir: Path, qlist: List[dict], seg_meta: List[dict]) -> None:
    """把 qlist 落盘：out_dir/queries/*.npy；同时写 manifest 与 seg_meta。"""
    qdir = out_dir / "queries"
    qdir.mkdir(parents=True, exist_ok=True)

    # tag+mirror 分桶
    buckets: Dict[Tuple[str, bool], List[np.ndarray]] = {}
    for item in qlist:  # {"tag","mirrored","vec"}
        key = (item["tag"], bool(item.get("mirrored", False)))
        buckets.setdefault(key, []).append(item["vec"].astype(np.float32))

    manifest = {"variants": [], "dim": int(qlist[0]["vec"].shape[-1]) if qlist else 0}
    for (tag, mir), vecs in buckets.items():
        arr = np.stack(vecs, axis=0) if vecs else np.zeros((0, manifest["dim"]), dtype=np.float32)
        fname = f"{tag}{'_m' if mir else ''}.npy"
        np.save(qdir / fname, arr)
        manifest["variants"].append({"file": fname, "tag": tag, "mirrored": mir, "count": int(arr.shape[0])})

    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta" / "seg_meta.json", "w", encoding="utf-8") as f:
        json.dump(seg_meta, f, ensure_ascii=False, indent=2)
    with open(out_dir / "meta" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser("recmatcher-encode-clip")
    ap.add_argument("--clip", required=True, help="输入短片 mp4")
    ap.add_argument("--clip_segs", required=False, help="分段 JSON [{'start':..,'end':..,'scene_id':..}, ...]")
    ap.add_argument("--out", required=True, help="输出目录（保存 .npy 查询向量）")
    ap.add_argument("--size", type=int, default=288, help="模型方形输入（默认 288）")
    ap.add_argument("--frames", type=int, default=12, help="每段采样帧数（默认 12）")
    ap.add_argument("--agg", default="gem", help="时序聚合（gem|mean），默认 gem")
    ap.add_argument("--include_mirror", action="store_true", help="同时保存镜像查询")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 打开短片
    clip_path = Path(args.clip)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open clip: {clip_path}")

    # 读取分段；如果没给，就整段做一个 segment
    segs = []
    if args.clip_segs:
        with open(args.clip_segs, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, s in enumerate(data):
            segs.append({
                "seg_idx": i,
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "scene_id": int(s.get("scene_id", -1)),
            })
    else:
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
        segs = [{"seg_idx": 0, "start": 0.0, "end": float(dur), "scene_id": -1}]

    # 查询端视角配置（统一编码，匹配时再选用）
    cfg = {
        "tight": True,
        "context": True,
        "horiz_offsets": ["left", "right"],  # 如需 center 可自行加上
        "mirrored": bool(args.include_mirror),
    }
    qr = QueryReadout(config=cfg, size=int(args.size), n_frames=3, agg=args.agg)
    _ = VPRemb()  # 触发一次初始化；QueryReadout 内部也会管理

    qlist_all: List[dict] = []
    seg_meta: List[dict] = []

    # 进度条：按 segment 粒度展示处理进度
    pbar = tqdm(segs, desc="Encoding segments", unit="seg")
    for s in pbar:
        frames = _read_frames_between(cap, s["start"], s["end"], n=args.frames)
        seg_meta.append(s)
        if not frames:
            continue
        # 产生该段的多视角向量
        qs = qr.make_queries(frames)  # [{"tag","mirrored","vec"}, ...]
        qs = [{"tag": q["tag"], "mirrored": q.get("mirrored", False), "vec": q["vec"].astype(np.float32)} for q in qs]
        qlist_all.extend(qs)
        # 更新进度条后缀信息
        try:
            pbar.set_postfix(frames=len(frames), queries=len(qs))
        except Exception:
            pass

    cap.release()

    # 落盘
    _save_queries(out_dir, qlist_all, seg_meta)

    info = {
        "clip_path": str(clip_path),
        "size": int(args.size),
        "frames_per_seg": int(args.frames),
        "agg": args.agg,
        "include_mirror": bool(args.include_mirror),
    }
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()