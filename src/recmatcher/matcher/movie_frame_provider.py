from __future__ import annotations
import cv2, numpy as np

class FFMPEGFrameProvider:
    """OpenCV-based sparse frame grabber (no heavy deps)."""
    def __init__(self):
        pass

    def get_frames(self, movie_path: str, t0: float, t1: float, fps: float = 3.0) -> np.ndarray:
        cap = cv2.VideoCapture(str(movie_path))
        n = max(2, int((t1 - t0) * fps))
        ts = np.linspace(t0, t1, n)
        frames = []
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000.0)
            ok, f = cap.read()
            if not ok: continue
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f.astype(np.float32)/255.0)
        cap.release()
        if not frames:
            return np.zeros((0,))
        return np.stack(frames, axis=0)
