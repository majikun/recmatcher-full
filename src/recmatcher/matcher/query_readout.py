from __future__ import annotations
from typing import List, Dict
import numpy as np
import cv2

def _square_center(img, size):
    h,w = img.shape[:2]
    if h < w:
        scale = size/h
    else:
        scale = size/w
    nw, nh = int(round(w*scale)), int(round(h*scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # center-crop to size
    x = max(0, (nw-size)//2); y = max(0, (nh-size)//2)
    crop = r[y:y+size, x:x+size]
    if crop.shape[0] != size or crop.shape[1] != size:
        canvas = np.zeros((size,size,3), dtype=r.dtype)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    return crop

def _square_letterbox(img, size):
    h, w = img.shape[:2]
    # scale so that the LONGER side becomes `size`
    scale = float(size) / float(max(h, w)) if max(h, w) > 0 else 1.0
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    nh = max(1, min(size, nh))
    nw = max(1, min(size, nw))

    # resize while keeping aspect ratio
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    r = cv2.resize(img, (nw, nh), interpolation=interp)

    # place centered on a square canvas
    canvas = np.zeros((size, size, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y:y+nh, x:x+nw] = r
    return canvas

def _square_hbias(img, size, side):
    h,w = img.shape[:2]
    if h < w:
        scale = size/h
    else:
        scale = size/w
    nw, nh = int(round(w*scale)), int(round(h*scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if nw <= size: return _square_center(img, size)
    if side == "left": x = 0
    elif side == "right": x = nw-size
    else: x = (nw-size)//2
    y = max(0,(nh-size)//2)
    crop = r[y:y+size, x:x+size]
    if crop.shape[0] != size or crop.shape[1] != size:
        canvas = np.zeros((size,size,3), dtype=r.dtype)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    return crop

class QueryReadout:
    """Produce multiple query patches from clip frames (tight/context + L/C/R + mirrored)."""
    def __init__(self, config: dict, size:int=288, n_frames:int=3, agg:str="gem"):
        self.cfg = config
        self.size = size
        self.n_frames = n_frames
        self.agg = agg

    def _agg(self, feats: np.ndarray) -> np.ndarray:
        # feats [F,D]
        if feats.ndim == 1: return feats
        if self.agg == "max": return feats.max(axis=0)
        if self.agg == "mean": return feats.mean(axis=0)
        # gem
        p=3.0
        return (np.mean(np.power(np.maximum(feats, 1e-6), p), axis=0))**(1.0/p)

    def _embed_imgs(self, imgs: List[np.ndarray]) -> np.ndarray:
        # lightweight RGB stats embedding to keep runnable
        arr = np.stack(imgs, axis=0).astype(np.float32)/255.0
        # per-frame embed: mean RGB + std + HSV 12-bin
        vecs=[]
        for a in arr:
            m = a.mean(axis=(0,1)); s = a.std(axis=(0,1))
            small = cv2.resize(a, (64,64))
            hsv = cv2.cvtColor((small*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            h = np.histogram(hsv[:,:,0], bins=12, range=(0,255))[0].astype(np.float32)
            h = h/(np.linalg.norm(h)+1e-6)
            v = np.concatenate([m,s,h], axis=0)
            vecs.append(v)
        V = np.stack(vecs, axis=0)
        return self._agg(V)

    def make_queries(self, clip_frames: List[np.ndarray]) -> List[Dict]:
        # select n frames evenly
        if not clip_frames: return []
        idx = np.linspace(0, len(clip_frames)-1, max(1,self.n_frames)).astype(int)
        frames = [clip_frames[i] for i in idx]
        patches = []
        if self.cfg.get("tight", True):
            patches.append(("tight", [_square_center(f, self.size) for f in frames]))
        if self.cfg.get("context", True):
            patches.append(("context", [_square_letterbox(f, self.size) for f in frames]))
        for side in self.cfg.get("horiz_offsets", ["left","center","right"]):
            patches.append((side, [_square_hbias(f, self.size, side) for f in frames]))

        out = []
        for tag, imgs in patches:
            vec = self._embed_imgs(imgs)
            out.append({"tag": tag, "vec": vec, "mirrored": False})
            if self.cfg.get("mirrored", True):
                imgs_m = [np.ascontiguousarray(img[:, ::-1, :]) for img in imgs]
                vecm = self._embed_imgs(imgs_m)
                out.append({"tag": tag, "vec": vecm, "mirrored": True})
        return out
