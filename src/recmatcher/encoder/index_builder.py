from __future__ import annotations
from pathlib import Path
import numpy as np
import faiss
from ..utils.faiss_utils import build_faiss_ip

class IndexBuilder:
    """Build FAISS indices for each variant, plus id_map."""
    def __init__(self, emb_dir: str|Path, index_dir: str|Path):
        self.emb_dir = Path(emb_dir); self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, variant: str, hnsw_m: int = 32) -> str:
        vec_path = self.emb_dir / f"{variant}.npy"
        idmap_path = self.emb_dir / f"{variant}_id_map.csv"
        if not vec_path.exists() or not idmap_path.exists():
            return ""
        vecs = np.load(vec_path).astype("float32")
        index = build_faiss_ip(vecs, hnsw_m=hnsw_m)
        out_path = self.index_dir / f"{variant}.faiss"
        faiss.write_index(index, str(out_path))
        return str(out_path)
