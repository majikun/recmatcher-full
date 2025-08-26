from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import math

@dataclass
class SceneConsensusParams:
    vote_top_n: int = 3
    dynamic_k: bool = True
    continuity_eps: float = 0.08        # near-tie margin for continuity tiebreak
    continuity_max_gap: int = 2         # prefer scenes enabling <= this gap for chains
    w_len: float = 0.3                  # penalty weight for |dur-1|
    w_mirror: float = 0.05              # mild penalty if mirrored_any
    w_consensus: float = 0.02           # per extra participating variant
    ratio_hi: float = 3.0               # if top1/top2 >= ratio_hi -> shrink K
    ratio_lo: float = 1.5               # if top1/top2 <= ratio_lo -> expand K

class SceneConsensus:
    """Decide one movie scene for a clip scene by aggregating evidence from seg-level candidates."""

    def __init__(self, params: SceneConsensusParams):
        self.p = params

    def _seg_topk(self, seg_info: Dict[str, Any]) -> int:
        K = int(self.p.vote_top_n)
        # dynamic K based on uncertainty ratio if available
        try:
            r = float(seg_info.get("uncertainty", {}).get("ratio", float("inf")))
        except Exception:
            r = float("inf")
        if not self.p.dynamic_k:
            return max(1, K)
        if r >= self.p.ratio_hi:
            return max(1, K - 1)
        if r <= self.p.ratio_lo:
            return K + 1
        return K

    def _cand_weight(self, cand: Dict[str, Any]) -> float:
        v = float(cand.get("vote", cand.get("score", 0.0)))
        # duration closeness
        dur_ratio = float(cand.get("duration_ratio", 1.0))
        pen = self.p.w_len * min(0.5, abs(dur_ratio - 1.0))
        v -= pen
        # mirrored penalty
        if bool(cand.get("mirrored_any", False)):
            v -= self.p.w_mirror
        # per-variant small consensus boost
        m = len(cand.get("source_variants") or [])
        if m > 1:
            v += self.p.w_consensus * (m - 1)
        return float(v)

    def aggregate(self, group: List[Dict[str, Any]], movie_by_scene: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        group: list of {'seg_id', 'candidates': [cand...], 'uncertainty': {...}} within one clip scene
        returns:
            {
              'scores': {scene_id: total_weight},
              'selected_scene': scene_id or None,
              'tie_break': str,
              'coverage': {scene_id: covered_count},
              'avg_len_dev': {scene_id: avg_abs_len_dev},
            }
        """
        scores: Dict[Optional[int], float] = {}
        covered: Dict[Optional[int], int] = {}
        len_dev_acc: Dict[Optional[int], float] = {}

        for seg in group:
            cands = list(seg.get("candidates") or [])
            if not cands:
                continue
            K = self._seg_topk(seg)
            for cand in cands[:max(1, K)]:
                sid = cand.get("scene_id")
                wt = self._cand_weight(cand)
                scores[sid] = scores.get(sid, 0.0) + wt
                covered[sid] = covered.get(sid, 0) + 1
                len_dev_acc[sid] = len_dev_acc.get(sid, 0.0) + abs(float(cand.get("duration_ratio", 1.0)) - 1.0)

        if not scores:
            return {"scores": {}, "selected_scene": None, "tie_break": "empty", "coverage": {}, "avg_len_dev": {}}

        # compute avg len dev
        avg_len_dev = {k: (len_dev_acc[k] / max(1, covered.get(k, 1))) for k in scores.keys()}

        # primary choice by score
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        sel, sel_score = ranked[0][0], ranked[0][1]

        # tie-break: if close to runner-up (within continuity_eps fraction), prefer the scene enabling longer chain / smaller gap
        reason = "score_max"
        if len(ranked) >= 2:
            s2, sc2 = ranked[1]
            if sel is not None and s2 is not None:
                if abs(sel_score - sc2) <= self.p.continuity_eps * max(1.0, sel_score):
                    # prefer better coverage; if tie, prefer smaller avg_len_dev
                    cov1 = covered.get(sel, 0); cov2 = covered.get(s2, 0)
                    if cov2 > cov1 + 1:  # much better coverage
                        sel, reason = s2, "tie_cov"
                    elif cov1 == cov2:
                        if avg_len_dev.get(s2, 9e9) + 1e-6 < avg_len_dev.get(sel, 9e9):
                            sel, reason = s2, "tie_len"
                        else:
                            reason = "tie_cov_len"
                    else:
                        reason = "tie_cov_pref"

        return {
            "scores": scores,
            "selected_scene": sel,
            "tie_break": reason,
            "coverage": covered,
            "avg_len_dev": avg_len_dev,
        }
