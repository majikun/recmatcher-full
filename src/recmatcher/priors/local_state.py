from __future__ import annotations
from collections import deque
from typing import Dict, Any, Optional

class LocalState:
    """
    Maintains light-weight priors for a sequential clip with possible global non-monotonic jumps:
      - Contiguity bonus: favor siblings (±1..2) within the same scene relative to last selected.
      - Mode prior: sliding window histogram over winning primary_variant.
    """

    def __init__(self, window: int = 10, w_cont: float = 0.12, w_mode: float = 0.2):
        self.window = int(max(1, window))
        self.w_cont = float(max(0.0, w_cont))
        self.w_mode = float(max(0.0, w_mode))
        self.recent_modes = deque(maxlen=self.window)
        self.prev_selected: Optional[Dict[str, Any]] = None

    # --- contiguity ---
    def contiguity_bonus(self, prev_cand: Optional[Dict[str, Any]], cand: Dict[str, Any]):
        if self.w_cont <= 0.0 or prev_cand is None or cand is None:
            return 0.0, {"bonus": 0.0, "is_same_scene": False, "is_sibling": False, "delta_seg": None}

        prev_scene = prev_cand.get("scene_id")
        scene = cand.get("scene_id")
        same_scene = (prev_scene is not None and scene is not None and prev_scene == scene)

        prev_idx = prev_cand.get("scene_seg_idx")
        idx = cand.get("scene_seg_idx")
        delta = None
        is_sibling = False
        bonus = 0.0

        if same_scene and prev_idx is not None and idx is not None:
            try:
                delta = int(idx) - int(prev_idx)
            except Exception:
                delta = None
            if delta in (-1, 1):
                bonus = 0.15 * self.w_cont
                is_sibling = True
            elif delta in (-2, 2):
                bonus = 0.08 * self.w_cont
                is_sibling = True
            else:
                bonus = 0.05 * self.w_cont  # same-scene but non-sibling
        elif same_scene:
            bonus = 0.05 * self.w_cont  # same-scene, missing indices
        else:
            bonus = 0.0

        return float(bonus), {
            "bonus": float(bonus),
            "is_same_scene": bool(same_scene),
            "is_sibling": bool(is_sibling),
            "delta_seg": delta,
        }

    # --- mode prior ---
    def update_mode_hist(self, selected_cand: Dict[str, Any]):
        mode = selected_cand.get("primary_variant")
        if mode:
            self.recent_modes.append(mode)

    def mode_prior_bonus(self, cand_mode: str):
        if self.w_mode <= 0.0 or not self.recent_modes:
            return 0.0, {"bonus": 0.0, "window": self.window, "P": {}}
        total = float(len(self.recent_modes))
        # frequency
        counts: Dict[str, int] = {}
        for m in self.recent_modes:
            counts[m] = counts.get(m, 0) + 1
        P = {k: (v / total) for k, v in counts.items()}
        bonus = self.w_mode * float(P.get(cand_mode, 0.0))
        return float(bonus), {"bonus": float(bonus), "window": self.window, "P": P}
