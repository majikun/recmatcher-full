from __future__ import annotations
from typing import List, Dict

def group_by_clip_scene(arr: List[dict]) -> Dict[int, List[dict]]:
    out: Dict[int,List[dict]] = {}
    for r in arr:
        out.setdefault(r.get("scene_id"), []).append(r)
    return out
