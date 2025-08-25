from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json
import numpy as np


def load_queries(qdir: str | Path) -> List[dict]:
    """
    读取 encode_clip.py 产出的查询向量，返回形如：
    [{"tag": str, "mirrored": bool, "vecs": np.ndarray [N, D]}, ...]
    """
    qdir = Path(qdir)
    manifest = json.loads((qdir / "meta" / "manifest.json").read_text())
    items: Dict[tuple, dict] = {}
    for v in manifest.get("variants", []):
        path = qdir / "queries" / v["file"]
        if path.exists():
            arr = np.load(path)
            key = (v["tag"], bool(v.get("mirrored", False)))
            items[key] = {
                "tag": v["tag"],
                "mirrored": bool(v.get("mirrored", False)),
            "vecs": arr.astype(np.float32, copy=False),
        }
    return list(items.values())


def save_queries_from_qlist(qdir: str | Path, qlist: List[dict]) -> None:
    """
    如果你已有内存里的 qlist（每个元素带 vecs [N,D]），想用同样布局保存，可用此工具。
    """
    from ..cli.encode_clip import _save_queries  # 复用保存逻辑

    out = []
    n = None
    for q in qlist:
        vecs = q["vecs"]
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if n is None:
            n = vecs.shape[0]
        for i in range(vecs.shape[0]):
            out.append({"tag": q["tag"], "mirrored": q.get("mirrored", False), "vec": vecs[i]})

    _save_queries(Path(qdir), out, [{"seg_idx": i, "start": 0.0, "end": 0.0, "scene_id": -1} for i in range(n or 0)])