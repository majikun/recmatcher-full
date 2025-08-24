from __future__ import annotations
import logging
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

    def search(self, x, topk: int):
        import numpy as _np
        # 统一为 float32 + C contiguous
        x = _np.asarray(x, dtype=_np.float32)
        if x.ndim == 1:
            x = x[None, :]
        # 清洗 NaN/Inf，避免底层崩溃
        if _np.isnan(x).any() or _np.isinf(x).any():
            logging.getLogger(__name__).warning(
                "[faiss] query contains NaN/Inf, sanitizing with nan_to_num()"
            )
            x = _np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = _np.ascontiguousarray(x, dtype=_np.float32)

        # 维度显式校验（用 Python 异常替代 C++ assert/segfault）
        if x.shape[1] != self.index.d:
            raise ValueError(
                f"Faiss dimension mismatch: query dim {x.shape[1]} vs index dim {self.index.d}"
            )
        # 真正检索
        return self.index.search(x, topk)

def build_faiss_ip(vecs: np.ndarray, hnsw_m: int = 32) -> faiss.Index:
    x = vecs.astype("float32")
    faiss.normalize_L2(x)
    index = faiss.IndexHNSWFlat(x.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(x)
    return index
