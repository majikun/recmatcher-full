from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

class IdMap:
    """CSV-based id mapping (faiss_id <-> metadata).

    Compatible APIs provided:
      - constructor: IdMap(path)
      - classmethods: from_csv(path), load(path), read_csv(path), from_path(path)
      - methods: get(i), __getitem__(i), __len__(), to_dict()
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path)
        # Ensure the underlying frame has a 0..N-1 index that matches FAISS row ids
        try:
            self.df = self.df.reset_index(drop=True)
        except Exception:
            pass

    # --- classmethod aliases for compatibility with different callers ---
    @classmethod
    def from_csv(cls, path: str | Path) -> "IdMap":
        return cls(path)

    @classmethod
    def load(cls, path: str | Path) -> "IdMap":
        return cls(path)

    @classmethod
    def read_csv(cls, path: str | Path) -> "IdMap":
        return cls(path)

    @classmethod
    def from_path(cls, path: str | Path) -> "IdMap":
        return cls(path)

    # --- access helpers ---------------------------------------------------
    def get(self, faiss_id: int) -> dict:
        row = self.df.iloc[int(faiss_id)].to_dict()
        return row

    def __getitem__(self, faiss_id: int) -> dict:
        return self.get(faiss_id)

    def __len__(self) -> int:
        return int(len(self.df))

    def to_dict(self) -> dict:
        """Return a mapping {row_id: row_dict} for convenience in code paths that expect a dict-like object."""
        return {int(i): rec for i, rec in self.df.reset_index(drop=True).to_dict(orient="index").items()}

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
