from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from ..types import Candidate

class SequenceAligner:
    """Monotonic alignment via DP: choose one candidate per segment with non-decreasing t0 maximizing fused score."""
    def align_monotonic(self, scene_plan: dict, seg2cands: Dict[int, List[Candidate]]) -> Dict[int, Candidate]:
        # assume seg IDs are sortable
        seg_ids = sorted(seg2cands.keys())
        # build lists
        cand_lists = [sorted(seg2cands[sid], key=lambda c: c.tw.t0) for sid in seg_ids]
        # DP
        paths: List[List[int]] = []
        scores: List[np.ndarray] = []
        for i, cl in enumerate(cand_lists):
            n = len(cl)
            if n == 0:
                scores.append(np.full((1,), -1e9)); paths.append([-1])
                continue
            sc = np.array([c.score_fused for c in cl], dtype=np.float32)
            if i == 0:
                scores.append(sc.copy()); paths.append([-1]*n)
            else:
                prev = scores[-1]; prev_cands = cand_lists[i-1]
                m = len(prev)
                new = np.full((n,), -1e9, dtype=np.float32)
                back = [-1]*n
                for j,c in enumerate(cl):
                    best = -1e9; bi = -1
                    for k in range(m):
                        if prev_cands[k].tw.t0 <= c.tw.t0:  # monotonic
                            val = prev[k] + sc[j]
                            if val > best:
                                best = val; bi = k
                    new[j] = best
                    back[j] = bi
                scores.append(new); paths.append(back)
        # backtrack
        result: Dict[int, Candidate] = {}
        if not scores: 
            return result
        last = len(scores)-1
        idx = int(np.argmax(scores[-1]))
        for i in reversed(range(len(cand_lists))):
            cl = cand_lists[i]
            if len(cl)==0: 
                continue
            result[seg_ids[i]] = cl[idx]
            idx = paths[i][idx] if paths[i][idx] >=0 else 0
        return result
