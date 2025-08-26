from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class SceneChainParams:
    max_skip: int = 1                 # allow skipping this many clip segs (soft)
    len_w: float = 0.3                # reward closeness of duration
    jump_penalty: float = 0.2         # penalty per index jump > 1
    strict_increase: bool = True      # **NEW**: enforce strictly increasing scene_seg_idx (no repeats)
    fill_enable: bool = False
    fill_max_gap: int = 2             # max size of hole to fill (in scene_seg_idx units)
    fill_penalty: float = 0.05        # small penalty for fills (confidence lower)

class SceneChain:
    """Greedy/heuristic longest chain builder inside one movie scene with optional *strictly increasing* idx (no repeats), then propose small-hole fills."""

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
        used_idx: set[int] = set()  # prevent duplicates even if we skip (prev_idx stays None)
        chain_len = 0

        for seg in group:
            sid = seg.get("seg_id")
            cands = [c for c in (seg.get("candidates") or []) if c.get("scene_id") == movie_scene_id]
            if not cands:
                assign[sid] = None
                continue

            # enforce strictly increasing idx (no repeats); allow small gaps up to max_skip
            allowed: List[Dict[str, Any]] = []
            for c in cands:
                idx = c.get("scene_seg_idx")
                if idx is None:
                    continue
                idx = int(idx)
                if self.p.strict_increase:
                    if prev_idx is not None:
                        d = idx - int(prev_idx)
                        if d <= 0:
                            continue  # no repeats or backtrack
                        if d > (self.p.max_skip + 1):
                            continue  # too large a jump inside a clip scene
                    # also avoid reusing the same idx when prev_idx didn't advance due to a miss
                    if idx in used_idx:
                        continue
                allowed.append(c)

            # if nothing allowed under strict rule, leave as None (gap) and keep prev_idx
            if self.p.strict_increase and not allowed:
                assign[sid] = None
                continue

            pool = allowed if allowed else cands
            # rank by continuity-aware reward within the feasible pool
            cands_ranked = sorted(pool, key=lambda c: -self._reward(prev_idx, c))
            best = cands_ranked[0]
            assign[sid] = best
            if best.get("scene_seg_idx") is not None:
                cur_idx = int(best["scene_seg_idx"])
                if prev_idx is None or cur_idx == int(prev_idx) + 1:
                    chain_len += 1
                prev_idx = cur_idx
                used_idx.add(cur_idx)

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

# -------------------------------
# Global (cross-clip-scene) time-chain DP with controlled flashbacks
# -------------------------------
@dataclass
class GlobalChainParams:
    alpha: float = 0.6   # penalty for time going backwards (flashback)
    beta: float = 0.2    # penalty for large forward jumps
    gamma: float = 0.2   # penalty for switching movie_scene frequently
    allow_flashbacks: int = 1  # how many backward moves are allowed globally
    topk: int = 3        # how many candidate scenes to consider per clip scene
    # New: scoring normalization & stickiness
    score_w: float = 1.0           # weight for (normalized) score term
    score_norm: str = "minmax"     # "minmax" | "softmax" | "none"
    softmax_tau: float = 0.5       # temperature for softmax normalization
    stick_bonus: float = 0.15      # reward (negative penalty) for staying in the same movie scene

class GlobalTimeChain:
    """
    Viterbi/DP over clip scenes. Each time step t corresponds to one clip_scene,
    with a small set of candidate movie scenes c[t] = [{scene_id, rep_time, score}, ...].
    We pick one candidate per step to minimize:
        cost = -score + alpha * backtrack + beta * big_forward_jump + gamma * scene_switch
    and allow a limited number of "flashbacks" (rep_time decreasing).

    Input 'series' is a list with length T (number of clip scenes). series[t] is a list of
    candidate dicts where each dict minimally contains:
      - 'scene_id': int
      - 'rep_time': float (representative time in seconds in the movie)
      - 'score': float (higher is better)
    """

    def __init__(self, params: GlobalChainParams):
        self.p = params

    def _trans_cost(self, prev: Dict[str, Any], cur: Dict[str, Any]) -> Tuple[float, int]:
        """
        Returns (transition_penalty, flashback_inc).
        """
        if prev is None:
            return 0.0, 0
        pen = 0.0
        fb_inc = 0
        tdiff = float(cur.get("rep_time", 0.0)) - float(prev.get("rep_time", 0.0))
        # time backtracking (flashback)
        if tdiff < 0:
            # constant + mild magnitude component so longer backtracks are a bit more penalized
            pen += self.p.alpha * (1.0 + min(3.0, -tdiff))
            fb_inc = 1
        # large forward jump (soft)
        if tdiff > 5.0:  # seconds threshold (heuristic)
            pen += self.p.beta * ((tdiff - 5.0) / 10.0)
        # scene switch penalty / stickiness reward
        if prev.get("scene_id") != cur.get("scene_id"):
            pen += self.p.gamma
        else:
            pen -= max(0.0, float(self.p.stick_bonus))
        return float(max(0.0, pen)), int(fb_inc)

    def plan(self, series: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        series: list over clip scenes; each item is a list of candidate dicts:
            { 'scene_id': int, 'rep_time': float, 'score': float, ... }
        Returns:
            {
              'choice_idx': [j0, j1, ...],        # index chosen within each series[t]
              'choice_scene': [scene_id, ...],    # chosen movie scene id
              'flashbacks': [t indices where backtrack happened],
              'cost': float
            }
        """
        if not series:
            return {'choice_idx': [], 'choice_scene': [], 'flashbacks': [], 'cost': 0.0}

        # Trim to topk per step (already scored descending)
        S: List[List[Dict[str, Any]]] = []
        for cands in series:
            c_sorted = sorted(cands, key=lambda x: -float(x.get("score", 0.0)))
            S.append(c_sorted[:max(1, int(self.p.topk))])

        # Per-step score normalization -> attach 'score_norm' used by DP
        for t in range(len(S)):
            vals = [float(c.get("score", 0.0)) for c in S[t]]
            if not vals:
                continue
            if self.p.score_norm == "softmax":
                import math
                tau = max(1e-3, float(self.p.softmax_tau))
                m = max(vals)
                exps = [math.exp((v - m) / tau) for v in vals]
                Z = sum(exps) + 1e-9
                for k, c in enumerate(S[t]):
                    c["score_norm"] = exps[k] / Z
            elif self.p.score_norm == "minmax":
                vmin, vmax = min(vals), max(vals)
                rng = max(1e-9, vmax - vmin)
                for c in S[t]:
                    c["score_norm"] = (float(c.get("score", 0.0)) - vmin) / rng
            else:
                # none: use raw score (not recommended if magnitude varies)
                for c in S[t]:
                    c["score_norm"] = float(c.get("score", 0.0))

        T = len(S)
        # dp[t][j][fb] = (cost, prev_j, prev_fb)
        INF = 1e18
        dp: List[List[List[Tuple[float, int, int]]]] = [
            [[(INF, -1, -1) for _ in range(self.p.allow_flashbacks + 1)] for _ in range(len(S[t]))]
            for t in range(T)
        ]

        # init
        for j, cand in enumerate(S[0]):
            base = -float(self.p.score_w) * float(cand.get("score_norm", cand.get("score", 0.0)))
            dp[0][j][0] = (base, -1, 0)

        # iterate
        for t in range(1, T):
            for j, cur in enumerate(S[t]):
                base = -float(self.p.score_w) * float(cur.get("score_norm", cur.get("score", 0.0)))
                best_cost, best_prev_j, best_prev_fb = INF, -1, -1
                for i, prev in enumerate(S[t-1]):
                    for fb_used in range(self.p.allow_flashbacks + 1):
                        prev_cost, _, _ = dp[t-1][i][fb_used]
                        if prev_cost >= INF:
                            continue
                        trans_pen, fb_inc = self._trans_cost(prev, cur)
                        new_fb = fb_used + fb_inc
                        if new_fb > self.p.allow_flashbacks:
                            continue
                        total = prev_cost + base + trans_pen
                        if total < best_cost:
                            best_cost, best_prev_j, best_prev_fb = total, i, new_fb
                dp[t][j][best_prev_fb if best_prev_fb >= 0 else 0] = (best_cost, best_prev_j, best_prev_fb)

        # backtrack best end state
        end_best = (INF, -1, -1)  # (cost, j, fb)
        t = T - 1
        for j in range(len(S[t])):
            for fb in range(self.p.allow_flashbacks + 1):
                c, _, _ = dp[t][j][fb]
                if c < end_best[0]:
                    end_best = (c, j, fb)

        choice_idx = [-1] * T
        fb_marks: List[int] = []
        cost, j, fb = end_best
        while t >= 0 and j >= 0:
            choice_idx[t] = j
            prev_cost, prev_j, prev_fb = dp[t][j][fb]
            # detect if this step consumed a flashback
            if t > 0 and prev_j >= 0:
                prev = S[t-1][prev_j]
                cur = S[t][j]
                tdiff = float(cur.get("rep_time", 0.0)) - float(prev.get("rep_time", 0.0))
                if tdiff < 0:
                    fb_marks.append(t)
            j = prev_j
            fb = prev_fb if prev_fb >= 0 else 0
            t -= 1

        choice_scene = [S[t][choice_idx[t]]["scene_id"] if choice_idx[t] >= 0 else None for t in range(T)]
        fb_marks.reverse()
        return {
            'choice_idx': choice_idx,
            'choice_scene': choice_scene,
            'flashbacks': fb_marks,
            'cost': float(cost),
        }
