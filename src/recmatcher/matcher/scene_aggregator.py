from __future__ import annotations
from typing import Dict, List
from collections import defaultdict
from ..types import Candidate

class SceneAggregator:
    """Simple vote: per clip_scene, pick movie_scene with max sum of fused scores."""
    def group_and_vote(self, seg2cands: Dict[int, List[Candidate]]) -> dict:
        scene_votes = defaultdict(lambda: defaultdict(float))
        for seg_id, cands in seg2cands.items():
            for c in cands[:5]:
                scene_votes[seg_id][c.tw.scene_id] += c.score_fused
        # choose best scene per seg
        seg_best_scene = {seg_id: max(v.items(), key=lambda kv: kv[1])[0] for seg_id, v in scene_votes.items() if v}
        return {"seg_best_scene": seg_best_scene}
