# src/recmatcher/cli/export_one_frame.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np

from ..encoder.preprocess import (
    to_square_letterbox,
    to_square_centercrop,
    to_square_anchor,
)
from ..types import CropVariant


def _duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    if fps <= 1e-6:
        return 0.0
    return frames / fps


def _read_middle_frame(video_path: Path, t0: float | None, t1: float | None) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    # 计算时间窗
    if t0 is None and t1 is None:
        dur = _duration_seconds(cap)
        t0, t1 = 0.0, dur
    elif t0 is not None and t1 is None:
        dur = _duration_seconds(cap)
        t1 = dur
    elif t0 is None and t1 is not None:
        t0 = 0.0
    # 取中点
    mid = float((min(t0, t1) + max(t0, t1)) / 2.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000.0)
    ok, f = cap.read()
    cap.release()
    if not ok or f is None:
        return None
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # RGB


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


def _save_rgb(img: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main():
    ap = argparse.ArgumentParser(description="Export ONE middle-frame image per angle for clip and movie.")
    ap.add_argument("--clip", required=True, help="短片路径")
    ap.add_argument("--movie", required=True, help="电影原片路径")
    ap.add_argument("--out", required=True, help="输出目录（将写入 clip/ 与 movie/ 子目录）")

    ap.add_argument("--variants", default="letterbox,centercrop,h_left,h_center,h_right",
                    help="导出的视角：逗号分隔")
    ap.add_argument("--size_base", type=int, default=288, help="letterbox/centercrop 尺寸（默认 288）")
    ap.add_argument("--size_anchor", type=int, default=144, help="h_left/center/right 尺寸（默认 144）")

    # 可选：只导某个时间窗的中间帧（单位：秒）
    ap.add_argument("--clip_t0", type=float, help="短片开始秒")
    ap.add_argument("--clip_t1", type=float, help="短片结束秒")
    ap.add_argument("--movie_t0", type=float, help="电影开始秒")
    ap.add_argument("--movie_t1", type=float, help="电影结束秒")

    ap.add_argument("--mirror", action="store_true", help="同时导出水平镜像（文件名加 _m 后缀）")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    movie_path = Path(args.movie)
    out_root = Path(args.out)

    # 读中间帧
    clip_frame = _read_middle_frame(clip_path, args.clip_t0, args.clip_t1)
    movie_frame = _read_middle_frame(movie_path, args.movie_t0, args.movie_t1)

    if clip_frame is None:
        raise SystemExit(f"Cannot read middle frame from clip: {clip_path}")
    if movie_frame is None:
        raise SystemExit(f"Cannot read middle frame from movie: {movie_path}")

    # 解析视角
    variant_list: list[CropVariant] = []
    for v in [s.strip() for s in str(args.variants).split(",") if s.strip()]:
        try:
            variant_list.append(CropVariant(v))
        except Exception:
            raise SystemExit(f"Unknown variant: {v}")
    if not variant_list:
        raise SystemExit("No variants specified")

    # 对 clip / movie 各自按视角导出一张
    for tag, frame in (("clip", clip_frame), ("movie", movie_frame)):
        for var in variant_list:
            is_anchor = var in (CropVariant.H_LEFT, CropVariant.H_CENTER, CropVariant.H_RIGHT)
            size = int(args.size_anchor) if is_anchor else int(args.size_base)
            out_img = _apply_variant(frame, var, size_base=int(args.size_base), size_anchor=int(args.size_anchor))
            _save_rgb(out_img, out_root / tag / f"{var.value}.jpg")
            if args.mirror:
                frame_m = np.ascontiguousarray(frame[:, ::-1, :])
                out_img_m = _apply_variant(frame_m, var, size_base=int(args.size_base), size_anchor=int(args.size_anchor))
                _save_rgb(out_img_m, out_root / tag / f"{var.value}_m.jpg")


if __name__ == "__main__":
    main()