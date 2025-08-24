from __future__ import annotations
from typing import List
import numpy as np
from ..types import TimeWindow

class FrameSampler:
    """Produce windows (win_len/stride) per segment, ensure last window covers the end."""
    def __init__(self, win_len: float = 2.0, stride: float = 2.0):
        self.win_len = float(win_len)
        self.stride = float(stride)

    def make_windows(self, movie_id: str, segs: list[dict]) -> List[TimeWindow]:
        tws: List[TimeWindow] = []
        for seg in segs:
            s = float(seg["start"]); e = float(seg["end"])
            scene_id = int(seg.get("scene_id", -1))
            if e - s <= 0: 
                continue
            win = self.win_len; st = self.stride if self.stride is not None else win
            if e - s <= win:
                tws.append(TimeWindow(movie_id, s, e, scene_id=scene_id))
                continue
            starts = list(np.arange(s, e - win + 1e-6, st))
            if starts and starts[-1] + win < e:
                starts.append(e - win)
            for t0 in starts:
                tws.append(TimeWindow(movie_id, float(t0), float(min(t0+win, e)), scene_id=scene_id))
        return tws
