from __future__ import annotations
from typing import List
import numpy as np
import cv2
from ..types import Candidate

def _htile_score(clip_patch: np.ndarray, cand_patch: np.ndarray, cols: int=8) -> float:
    """Compute horizontal tile similarity: cut into vertical columns and take max cosine over columns."""
    # clip_patch, cand_patch: [H,W,3] float32 0-1
    H,W,_ = clip_patch.shape
    clip_cols = np.array_split(clip_patch, cols, axis=1)
    cand_cols = np.array_split(cand_patch, cols, axis=1)
    def fe(img):
        v = np.concatenate([img.mean(axis=(0,1)), img.std(axis=(0,1))])
        # add grayscale histogram
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        hist = np.histogram(gray, bins=16, range=(0,255))[0].astype(np.float32)
        hist /= (np.linalg.norm(hist)+1e-6)
        return np.concatenate([v, hist])
    clip_feats = [fe(c) for c in clip_cols]
    cand_feats = [fe(c) for c in cand_cols]
    # cosine matrix
    C = np.zeros((len(clip_feats), len(cand_feats)), dtype=np.float32)
    for i,a in enumerate(clip_feats):
        a = a / (np.linalg.norm(a)+1e-6)
        for j,b in enumerate(cand_feats):
            b = b / (np.linalg.norm(b)+1e-6)
            C[i,j] = float(np.dot(a,b))
    return float(C.max())

def _temporal_energy(frames: np.ndarray) -> float:
    if frames.shape[0] < 2:
        return 0.0
    diff = np.abs(np.diff(frames.mean(axis=(2,3)), axis=0)).mean()
    return float(diff)

class Stage2Reranker:
    def __init__(self, config: dict):
        self.cfg = config

    def score(self, clip_frames: np.ndarray, cands: List[Candidate], movie_reader) -> List[Candidate]:
        if not cands:
            return cands
        # make clip reference patches: use center square as proxy
        ref = clip_frames.mean(axis=0)  # [H,W,3]
        ref = cv2.resize(ref, (288,288), interpolation=cv2.INTER_AREA)
        results = []
        for c in cands:
            # fetch candidate frames (sparse) and average to a patch
            frames = movie_reader.get_frames(c.tw.movie_id_path if hasattr(c.tw, "movie_id_path") else c.tw.movie_id, c.tw.t0, c.tw.t1, fps=3.0)
            if frames.ndim < 3:
                c.score_tile = 0.0; c.score_dyn = 0.0; c.score_cut = 0.0; c.score_prior = 0.0
            else:
                patch = cv2.resize(frames.mean(axis=0), (288,288))
                c.score_tile = _htile_score(ref, patch, cols=self.cfg.get("htile",{}).get("cols",8))
                c.score_dyn  = _temporal_energy(frames)
                c.score_cut  = 0.0  # if near_cut info available, add here
                c.score_prior= 0.0  # time prior can be injected by caller
            # fused
            w = self.cfg.get("weights", {"vp":0.45,"tile":0.35,"dynamic":0.15,"cut":0.05,"prior":0.02})
            c.score_fused = (w["vp"]*c.score_vp + w["tile"]*c.score_tile + w["dynamic"]*c.score_dyn +
                             w["cut"]*c.score_cut + w["prior"]*c.score_prior)
            results.append(c)
        return sorted(results, key=lambda x: -x.score_fused)
