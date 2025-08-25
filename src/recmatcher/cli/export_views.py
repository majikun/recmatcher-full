from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

from ..encoder.preprocess import (
    to_square_letterbox,
    to_square_centercrop,
    to_square_anchor,
)
from ..types import CropVariant


def _read_frames_between(cap: cv2.VideoCapture, t0: float, t1: float, n: int) -> List[np.ndarray]:
    if t1 < t0:
        t0, t1 = t1, t0
    ts = np.linspace(float(t0), float(t1), max(2, int(n)))
    frames: List[np.ndarray] = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, f = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


def _segments_from_args(cap: cv2.VideoCapture, clip_segs: str | None, t0: float | None, t1: float | None) -> List[Dict]:
    if clip_segs:
        with open(clip_segs, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = []
        for i, s in enumerate(data):
            segs.append({
                "seg_idx": i,
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "scene_id": int(s.get("scene_id", -1)),
            })
        return segs
    if t0 is not None or t1 is not None:
        t0 = float(t0 or 0.0)
        if t1 is None:
            # use end of video
            fps = max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
            dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            t1 = float(dur)
        return [{"seg_idx": 0, "start": float(min(t0, t1)), "end": float(max(t0, t1)), "scene_id": -1}]
    # whole clip as single segment
    fps = max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    return [{"seg_idx": 0, "start": 0.0, "end": float(dur), "scene_id": -1}]


def _apply_variant(img: np.ndarray, variant: CropVariant, size_base: int, size_anchor: int) -> np.ndarray:
    if variant == CropVariant.LETTERBOX:
        out, _ = to_square_letterbox(img, size_base)
        return out
    if variant == CropVariant.CENTERCROP:
        out, _ = to_square_centercrop(img, size_base)
        return out
    if variant == CropVariant.H_LEFT:
        out, _ = to_square_anchor(img, size_anchor, "left")
        return out
    if variant == CropVariant.H_CENTER:
        out, _ = to_square_anchor(img, size_anchor, "center")
        return out
    if variant == CropVariant.H_RIGHT:
        out, _ = to_square_anchor(img, size_anchor, "right")
        return out
    raise ValueError(f"Unknown variant: {variant}")


def main():
    ap = argparse.ArgumentParser(description="Export cropped views (angles) from a video segment list.")
    ap.add_argument("--video", required=True, help="输入视频文件路径")
    ap.add_argument("--out", required=True, help="输出目录（将创建子目录按 variant 分类）")
    ap.add_argument("--segs", help="分段 JSON [{'start':..,'end':..,'scene_id':..}]；不提供则整段/或用 --t0/--t1")
    ap.add_argument("--t0", type=float, help="可选：起始秒，与 --t1 搭配或单独使用")
    ap.add_argument("--t1", type=float, help="可选：结束秒，不给则默认为视频结束")

    ap.add_argument("--variants", default="letterbox,centercrop,h_left,h_center,h_right",
                    help="要导出的视角，逗号分隔：letterbox,centercrop,h_left,h_center,h_right")
    ap.add_argument("--size_base", type=int, default=288, help="letterbox/centercrop 的方形尺寸，默认 288")
    ap.add_argument("--size_anchor", type=int, default=144, help="h_left/center/right 的方形尺寸，默认 144")

    ap.add_argument("--frames_base", type=int, default=12, help="每段采样帧数（base 视角），默认 12")
    ap.add_argument("--frames_anchor", type=int, default=8, help="每段采样帧数（anchor 视角），默认 8")

    ap.add_argument("--mirror", action="store_true", help="同时导出水平镜像图像（以 _m 后缀命名）")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    segs = _segments_from_args(cap, args.segs, args.t0, args.t1)

    # parse variants
    variant_list: List[CropVariant] = []
    for v in [s.strip() for s in str(args.variants).split(',') if s.strip()]:
        try:
            variant_list.append(CropVariant(v))
        except Exception:
            raise SystemExit(f"Unknown variant: {v}")
    if not variant_list:
        raise SystemExit("No variants specified")

    # prepare output directories
    for v in variant_list:
        (out_dir / v.value).mkdir(parents=True, exist_ok=True)
        if args.mirror:
            (out_dir / f"{v.value}_m").mkdir(parents=True, exist_ok=True)

    # iterate segments and export
    pbar = tqdm(segs, desc="Exporting views", unit="seg")
    for s in pbar:
        t0 = float(s["start"]) ; t1 = float(s["end"]) ; seg_idx = int(s.get("seg_idx", 0))
        # read frames once per segment at max needed count
        n_max = max(int(args.frames_base), int(args.frames_anchor))
        frames = _read_frames_between(cap, t0, t1, n=n_max)
        if not frames:
            continue

        for v in variant_list:
            is_anchor = v in (CropVariant.H_LEFT, CropVariant.H_CENTER, CropVariant.H_RIGHT)
            n_use = int(args.frames_anchor) if is_anchor else int(args.frames_base)
            # choose evenly spaced subset from frames
            idx = np.linspace(0, len(frames)-1, max(1, n_use)).astype(int)
            for j, i in enumerate(idx):
                rgb = frames[int(i)]
                out_img = _apply_variant(rgb, v, size_base=int(args.size_base), size_anchor=int(args.size_anchor))

                # write RGB -> BGR for cv2.imwrite
                bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                fname = f"seg{seg_idx:04d}_f{j:03d}.jpg"
                cv2.imwrite(str(out_dir / v.value / fname), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                if args.mirror:
                    rgb_m = np.ascontiguousarray(rgb[:, ::-1, :])
                    out_img_m = _apply_variant(rgb_m, v, size_base=int(args.size_base), size_anchor=int(args.size_anchor))
                    bgr_m = cv2.cvtColor(out_img_m, cv2.COLOR_RGB2BGR)
                    fname_m = f"seg{seg_idx:04d}_f{j:03d}.jpg"
                    cv2.imwrite(str(out_dir / f"{v.value}_m" / fname_m), bgr_m, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        pbar.set_postfix(seg=seg_idx, frames=len(frames))

    cap.release()


if __name__ == "__main__":
    main()
