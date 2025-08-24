from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

class IdMap:
    """CSV-based id mapping (faiss_id <-> metadata)."""
    def __init__(self, path: str|Path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path)

    def get(self, faiss_id: int) -> dict:
        row = self.df.iloc[faiss_id].to_dict()
        return row

class FaissIndex:
    def __init__(self, index_path: str|Path, normalize: bool = True):
        self.index_path = str(index_path)
        self.normalize = normalize
        self.index = faiss.read_index(self.index_path)
        self.d = self.index.d

    def search(self, vecs: np.ndarray, topk: int):
        x = np.asarray(vecs, dtype="float32")
        if x.ndim == 1:
            x = x[None, :]
        # sanity check: dim must match index dim
        if x.shape[1] != self.index.d:
            raise ValueError(
                f"FAISS dim mismatch: query vectors have dim={x.shape[1]}, "
                f"but index dim={self.index.d}. This usually means the VPR model/variant "
                f"used at query time differs from the one used to build the movie index.\n"
                f"→ Fix: ensure the same VideoPrism backend/model is used on both sides, and the preprocess is identical."
            )
        if self.normalize:
            faiss.normalize_L2(x)
        D, I = self.index.search(x, topk)
        return D, I

def build_faiss_ip(vecs: np.ndarray, hnsw_m: int = 32) -> faiss.Index:
    x = vecs.astype("float32")
    faiss.normalize_L2(x)
    index = faiss.IndexHNSWFlat(x.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(x)
    return index
