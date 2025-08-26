from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class SceneChainParams:
    max_skip: int = 1                 # allow skipping this many clip segs (soft)
    len_w: float = 0.3                # reward closeness of duration
    jump_penalty: float = 0.2         # penalty per index jump > 1
    fill_enable: bool = False
    fill_max_gap: int = 2             # max size of hole to fill (in scene_seg_idx units)
    fill_penalty: float = 0.05        # small penalty for fills (confidence lower)

class SceneChain:
    """Greedy/heuristic longest chain builder inside one movie scene, then propose small-hole fills."""

    def __init__(self, params: SceneChainParams):
        self.p = params

    def _reward(self, prev_idx: Optional[int], cand: Dict[str, Any]) -> float:
        v = float(cand.get("vote", cand.get("score", 0.0)))
        # duration closeness reward
        dr = abs(float(cand.get("duration_ratio", 1.0)) - 1.0)
        v += self.p.len_w * (1.0 - min(0.5, dr))
        # continuity shaping
        idx = cand.get("scene_seg_idx")
        if prev_idx is not None and idx is not None:
            d = int(idx) - int(prev_idx)
            if d == 1:
                v += 0.15
            elif d == 2:
                v += 0.06
            elif d <= 0 or d > 2:
                v -= self.p.jump_penalty * (abs(d) if d > 0 else 1.5)
        return float(v)

    def build(self, group: List[Dict[str, Any]], movie_scene_id: int, movie_by_scene: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        group: [{'seg_id', 'candidates': [cand...]}] for one clip scene, in clip order
        Returns { 'assign': {seg_id: chosen_cand or None}, 'fills': [fill_obj...], 'stats': {...} }
        """
        assign: Dict[int, Optional[Dict[str, Any]]] = {}
        prev_idx: Optional[int] = None
        chain_len = 0

        for seg in group:
            sid = seg.get("seg_id")
            cands = [c for c in (seg.get("candidates") or []) if c.get("scene_id") == movie_scene_id]
            if not cands:
                assign[sid] = None
                continue
            # rank by continuity-aware reward
            cands_ranked = sorted(cands, key=lambda c: -self._reward(prev_idx, c))
            best = cands_ranked[0]
            assign[sid] = best
            if best.get("scene_seg_idx") is not None:
                if prev_idx is None or int(best["scene_seg_idx"]) == int(prev_idx) + 1:
                    chain_len += 1
                prev_idx = int(best["scene_seg_idx"]) if best.get("scene_seg_idx") is not None else prev_idx

        # propose fills (optional)
        fills: List[Dict[str, Any]] = []
        if self.p.fill_enable and movie_scene_id in movie_by_scene:
            seq = [r for r in (movie_by_scene.get(movie_scene_id) or []) if r.get("scene_seg_idx") is not None]
            seq = sorted(seq, key=lambda r: r.get("scene_seg_idx"))
            # collect selected idxes
            chosen_idx = [int(v.get("scene_seg_idx")) for v in assign.values() if v and v.get("scene_seg_idx") is not None]
            if chosen_idx:
                lo, hi = min(chosen_idx), max(chosen_idx)
                have = set(chosen_idx)
                for idx in range(lo, hi + 1):
                    if idx in have:
                        continue
                    # small hole only
                    if len([x for x in range(idx, idx + self.p.fill_max_gap + 1) if x not in have]) > self.p.fill_max_gap:
                        continue
                    # find the movie seg metadata with this idx
                    cand = next((r for r in seq if int(r.get("scene_seg_idx")) == idx), None)
                    if cand:
                        fills.append({
                            "type": "fill",
                            "scene_seg_idx": int(idx),
                            "start": float(cand.get("start", 0.0)),
                            "end": float(cand.get("end", 0.0)),
                            "scene_id": movie_scene_id,
                            "penalty": float(self.p.fill_penalty),
                        })

        stats = {"chain_len": chain_len, "selected": sum(1 for v in assign.values() if v is not None)}
        return {"assign": assign, "fills": fills, "stats": stats}
