from __future__ import annotations
from typing import Iterable, List
from pathlib import Path
import numpy as np
import pandas as pd
from ..types import EmbeddingRecord, CropVariant

class EmbStore:
    """Persist embeddings and metadata to disk (npz + id_map.csv)."""
    def __init__(self, base_dir: str|Path):
        self.base = Path(base_dir)
        (self.base / "emb").mkdir(parents=True, exist_ok=True)
        (self.base / "index").mkdir(parents=True, exist_ok=True)
        (self.base / "meta").mkdir(parents=True, exist_ok=True)
        self._buffers = {v: [] for v in CropVariant}

    def append(self, variant: str, records: Iterable[EmbeddingRecord]) -> None:
        self._buffers[CropVariant(variant)].extend(list(records))

    def finalize_variant(self, variant: CropVariant) -> str:
        recs: List[EmbeddingRecord] = self._buffers.get(variant, [])
        if not recs:
            return ""
        vecs = np.stack([r.vec.astype(np.float32) for r in recs], axis=0)
        np.save(self.base / "emb" / f"{variant.value}.npy", vecs.astype(np.float32))
        # id_map
        rows = []
        for i, r in enumerate(recs):
            rows.append({
                "faiss_id": i,
                "movie_id": r.tw.movie_id,
                "t0": r.tw.t0,
                "t1": r.tw.t1,
                "scene_id": r.tw.scene_id,
                "shot_id": r.tw.shot_id,
                "variant": r.variant.value,
                "bbox_x1": r.bbox_norm[0],
                "bbox_y1": r.bbox_norm[1],
                "bbox_x2": r.bbox_norm[2],
                "bbox_y2": r.bbox_norm[3],
                "size": r.size,
                "n_frames": r.n_frames,
                "stride_s": r.stride_s
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(self.base / "emb" / f"{variant.value}_id_map.csv", index=False)
        return str(self.base / "emb" / f"{variant.value}.npy")

    def finalize(self) -> None:
        for v in CropVariant:
            self.finalize_variant(v)
