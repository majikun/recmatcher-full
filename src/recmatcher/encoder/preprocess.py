from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2

def _resize_to_fit(frame: np.ndarray, target_max: int):
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("invalid frame")
    scale = target_max / max(h, w)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    # clamp to avoid rare rounding to size+1 and to keep positive sizes
    nw = max(1, min(target_max, nw))
    nh = max(1, min(target_max, nh))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    return resized, scale

def _resize_keep_ar(frame: np.ndarray, target_short: int):
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("invalid frame")
    scale = (target_short / h) if h < w else (target_short / w)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    nw = max(1, nw)
    nh = max(1, nh)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    return resized, scale

def to_square_letterbox(frame: np.ndarray, size: int) -> Tuple[np.ndarray, tuple]:
    # For letterbox we must fit the entire image inside the square -> scale by long side
    img, scale = _resize_to_fit(frame, size)
    h, w = img.shape[:2]
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    x = max(0, (size - w) // 2)
    y = max(0, (size - h) // 2)
    canvas[y:y+h, x:x+w] = img
    # bbox of the content area within the square (normalized)
    x1 = x / size; y1 = y / size; x2 = (x + w) / size; y2 = (y + h) / size
    return canvas, (x1, y1, x2, y2)

def to_square_centercrop(frame: np.ndarray, size: int) -> Tuple[np.ndarray, tuple]:
    img, scale = _resize_keep_ar(frame, size)
    nh, nw = img.shape[:2]
    # crop center to size x size
    x = max(0, (nw - size) // 2); y = max(0, (nh - size) // 2)
    x2 = min(nw, x + size); y2 = min(nh, y + size)
    crop = img[y:y2, x:x2]
    if crop.shape[0] != size or crop.shape[1] != size:
        # pad if needed
        canvas = np.zeros((size, size, 3), dtype=img.dtype)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    # bbox relative to resized image then normalized to square
    # here bbox is exactly the crop area placed at (0,0) in square
    return crop, (0.0, 0.0, 1.0, 1.0)

def to_square_anchor(frame: np.ndarray, size: int, anchor: str) -> Tuple[np.ndarray, tuple]:
    img, scale = _resize_keep_ar(frame, size)
    nh, nw = img.shape[:2]
    # horizontal bias: left/center/right
    if nw <= size:
        # not enough width to bias; fallback to center crop
        return to_square_centercrop(frame, size)
    if anchor == "left":
        x = 0
    elif anchor == "right":
        x = max(0, nw - size)
    else:
        x = max(0, (nw - size)//2)
    y = max(0, (nh - size)//2)
    x2 = min(nw, x+size); y2 = min(nh, y+size)
    crop = img[y:y2, x:x2]
    if crop.shape[0] != size or crop.shape[1] != size:
        canvas = np.zeros((size, size, 3), dtype=img.dtype)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    # bbox in square normalized relative to resized image
    # map crop location inside the resized image to square canvas at (0,0)
    # we report bbox relative to resized image then projected to square: use (0,0,1,1)
    return crop, (x/nw, y/nh, (x+size)/nw, (y+size)/nh)
