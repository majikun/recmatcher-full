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
        import faiss as _faiss
        # Normalize input: 2D, float32, contiguous
        x = _np.asarray(x, dtype=_np.float32)
        if x.ndim == 1:
            x = x[None, :]
        # Scrub NaN/Inf early to avoid C++ crashes
        if _np.isnan(x).any() or _np.isinf(x).any():
            logging.getLogger(__name__).warning(
                "[faiss] query contains NaN/Inf, sanitizing with nan_to_num()"
            )
            x = _np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = _np.ascontiguousarray(x, dtype=_np.float32)

        # Explicit dim check to prevent C++ side assertion/segfault
        d = getattr(self.index, 'd', None)
        if d is None:
            raise RuntimeError("Faiss index not initialized or missing 'd' attribute")
        if x.shape[1] != d:
            raise ValueError(f"Faiss dimension mismatch: query dim {x.shape[1]} vs index dim {d}")

        # Force single-thread to avoid Accelerate/OpenMP crashes on macOS/arm64
        try:
            _faiss.omp_set_num_threads(1)
        except Exception:
            pass

        return self.index.search(x, topk)
     
def build_faiss_ip(vecs: np.ndarray, hnsw_m: int = 32) -> faiss.Index:
    x = vecs.astype("float32")
    faiss.normalize_L2(x)
    index = faiss.IndexHNSWFlat(x.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(x)
    return index
