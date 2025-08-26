from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import math

@dataclass
class FusionParams:
    tau: float = 0.8            # temperature for soft vote
    z_clip: float = 5.0         # clip standardized scores to [-z_clip, z_clip]
    var_weights: Dict[str, float] = None
    mirror_penalty: float = 0.97
    w_consensus: float = 0.05   # small bonus per additional participating variant
    w_len: float = 0.2          # duration consistency penalty weight
    topk: int = 50              # default local topk cap per variant

    def __post_init__(self):
        if self.var_weights is None:
            self.var_weights = {}

class ScoreFusion:
    """
    Variant-level fusion with robust calibration and explainability.
    For each seg, we:
      1) Standardize per-variant scores by robust stats (median/IQR) -> z, then clip.
      2) Convert to soft-votes with temperature tau and variant/mirror weights.
      3) Aggregate votes by movie key (start,end,scene).
      4) Add small consensus bonus based on participating variants.
      5) Apply a light duration consistency penalty.
    Returns candidate dicts ready for downstream re-ranking / UI, with rich explain fields.
    """

    def __init__(self, params: FusionParams):
        self.p = params

    @staticmethod
    def _row_start_end(m: Dict[str, Any]):
        s = m.get("start")
        e = m.get("end")
        if s is None:
            s = m.get("t0", 0.0)
        if e is None:
            e = m.get("t1", 0.0)
        return float(s or 0.0), float(e or 0.0)

    def fuse_segment(
        self,
        seg_id: int,
        bag: Dict[str, List[Dict[str, Any]]],
        retriever,
        id_rows: Dict[str, List[Dict[str, Any]]],
        calib_tbl: Dict[str, Tuple[float, float]],
        clip_dur: float,
        topk: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        bag: {variant(str): [ {"vec": ndarray[D], "mirrored": bool}, ... ]}
        id_rows: {variant(str): [ rowdict, ... ]} with rowdict including start/end/scene_id/seg_id/scene_seg_idx
        calib_tbl: {variant: (median, iqr)}
        """
        K = int(topk or self.p.topk)

        fuse_vote = {}            # key -> total vote
        fuse_meta = {}            # key -> representative meta row (from the highest raw score)
        fuse_best = {}            # key -> best raw score among hits
        # explain aggregates
        fuse_var_votes_plain = {}   # key -> {variant: vote_sum_plain}
        fuse_var_votes_mirror = {}  # key -> {variant: vote_sum_mirror}
        fuse_any_mirrored = {}      # key -> bool

        for var, vec_list in (bag or {}).items():
            if not vec_list:
                continue
            import numpy as _np
            X = _np.stack([it["vec"] for it in vec_list]).astype(_np.float32)
            mir_flags = [bool(it.get("mirrored", False)) for it in vec_list]
            try:
                D, I, _idm = retriever._search_variant(var, X)
            except Exception:
                continue
            rows = id_rows.get(var, [])
            mu, iqr = calib_tbl.get(var, (0.0, 0.06))
            iqr = float(max(iqr, 1e-3))
            w_var = float(self.p.var_weights.get(var, 1.0))
            for r in range(I.shape[0] if I is not None else 0):
                mir_w_base = (self.p.mirror_penalty if mir_flags[r] else 1.0)
                for k in range(min(I.shape[1], K)):
                    j = int(I[r, k])
                    if j < 0 or j >= len(rows):
                        continue
                    m = rows[j]
                    ss, ee = self._row_start_end(m)
                    key = (ss, ee, m.get("scene_id"))
                    sc = float(D[r, k])

                    # robust standardization + clip
                    z = (sc - mu) / iqr
                    if z > self.p.z_clip:
                        z = self.p.z_clip
                    elif z < -self.p.z_clip:
                        z = -self.p.z_clip

                    # soft-vote with temperature, variant & mirror weights
                    base_vote = math.exp(z / max(self.p.tau, 1e-6)) * w_var
                    vote_plain = base_vote
                    vote_mirror = base_vote * mir_w_base  # if mirrored, apply penalty

                    # accumulate
                    total_vote = vote_mirror if mir_flags[r] else vote_plain
                    fuse_vote[key] = fuse_vote.get(key, 0.0) + total_vote

                    if mir_flags[r]:
                        if key not in fuse_var_votes_mirror:
                            fuse_var_votes_mirror[key] = {}
                        fuse_var_votes_mirror[key][var] = fuse_var_votes_mirror[key].get(var, 0.0) + float(vote_mirror)
                        fuse_any_mirrored[key] = True
                    else:
                        if key not in fuse_var_votes_plain:
                            fuse_var_votes_plain[key] = {}
                        fuse_var_votes_plain[key][var] = fuse_var_votes_plain[key].get(var, 0.0) + float(vote_plain)
                        if key not in fuse_any_mirrored:
                            fuse_any_mirrored[key] = False

                    # record best raw score & meta
                    if key not in fuse_best or sc > fuse_best[key]:
                        fuse_best[key] = sc
                        fuse_meta[key] = m

        # Assemble candidates
        items: List[Dict[str, Any]] = []
        eps = 1e-9
        for key, vt in fuse_vote.items():
            m = fuse_meta.get(key)
            if not m:
                continue
            ss, ee = self._row_start_end(m)
            dur = max(ee - ss, eps)
            dur_ratio = (dur / max(clip_dur, eps)) if clip_dur > eps else 1.0

            var_plain = fuse_var_votes_plain.get(key, {}) or {}
            var_mirror = fuse_var_votes_mirror.get(key, {}) or {}
            # sum votes per variant across plain+mirror
            var_total = {vk: float(var_plain.get(vk, 0.0) + var_mirror.get(vk, 0.0))
                         for vk in set(list(var_plain.keys()) + list(var_mirror.keys()))}
            source_variants = sorted(list(var_total.keys()))
            m_participate = len(source_variants)

            vote_final = float(vt)
            reason_codes = []

            # consensus bonus
            if m_participate > 1 and self.p.w_consensus > 0:
                bonus = self.p.w_consensus * float(m_participate - 1)
                vote_final += bonus
                reason_codes.append("consensus+")

            # duration consistency penalty
            if self.p.w_len > 0:
                pen = min(0.3, abs(dur_ratio - 1.0))
                if pen > 0.0:
                    vote_final -= self.p.w_len * pen
                    reason_codes.append("len-")

            # primary variant for mode summary
            primary_variant = None
            if var_total:
                primary_variant = max(var_total.items(), key=lambda kv: kv[1])[0]

            cand = {
                "seg_id": m.get("seg_id"),
                "scene_seg_idx": m.get("scene_seg_idx"),
                "start": ss,
                "end": ee,
                "scene_id": m.get("scene_id"),
                "score": float(fuse_best.get(key, 0.0)),  # raw score (vp) for observation
                "vote": float(vote_final),                # fused score for ranking
                "faiss_id": m.get("faiss_id"),
                "movie_id": m.get("movie_id"),
                "shot_id": m.get("shot_id"),
                # explain extras
                "variant_votes_plain": {vk: float(vv) for vk, vv in var_plain.items()},
                "variant_votes_mirror": {vk: float(vv) for vk, vv in var_mirror.items()},
                "variant_votes": var_total,
                "source_variants": source_variants,
                "mirrored_any": bool(fuse_any_mirrored.get(key, False)),
                "duration_ratio": float(dur_ratio),
                "reason_codes": reason_codes,
                "primary_variant": primary_variant,
            }
            items.append(cand)

        items.sort(key=lambda x: (-x.get("vote", x.get("score", 0.0)), -x.get("score", 0.0)))
        return items[:K]
