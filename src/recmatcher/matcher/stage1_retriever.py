from __future__ import annotations
from typing import List, Dict
import numpy as np
from ..types import Candidate, CropVariant, TimeWindow
from ..utils.faiss_utils import FaissIndex, IdMap

class Stage1Retriever:
    def __init__(self, index_paths: Dict[CropVariant,str], id_maps: Dict[CropVariant,IdMap], topk:int=80):
        self.index_paths = index_paths
        self.id_maps = id_maps
        self.topk = topk
        self._indices = {v: (FaissIndex(p) if p else None) for v,p in index_paths.items()}

    def _search_variant(self, variant: CropVariant, qvecs: np.ndarray):
        fi = self._indices.get(variant)
        if fi is None: 
            return None, None, None
        D, I = fi.search(qvecs, self.topk)
        idmap = self.id_maps[variant]
        return D, I, idmap

    def search(self, queries: List[dict],
               prefer_variants = [CropVariant.LETTERBOX, CropVariant.CENTERCROP],
               fallback_variants = [CropVariant.H_LEFT, CropVariant.H_RIGHT, CropVariant.H_CENTER]) -> List[Candidate]:
        # stack all query vecs
        q = np.stack([q["vec"] for q in queries]).astype(np.float32)
        # normalize to cosine
        q = q / (np.linalg.norm(q, axis=1, keepdims=True)+1e-6)
        cands: List[Candidate] = []
        tried = []
        for stage in (prefer_variants, fallback_variants):
            for var in stage:
                D, I, idmap = self._search_variant(var, q)
                if D is None: 
                    continue
                for qi in range(q.shape[0]):
                    for rank in range(I.shape[1]):
                        idx = int(I[qi, rank])
                        score = float(D[qi, rank])
                        meta = idmap.get(idx)
                        tw = TimeWindow(meta["movie_id"], float(meta["t0"]), float(meta["t1"]), int(meta["scene_id"]), int(meta["shot_id"]))
                        cand = Candidate(tw=tw, variant=var, score_vp=score, source_index=var.value, source_id=idx, explain={"q_tag": queries[qi]["tag"], "mirrored": queries[qi]["mirrored"]})
                        cands.append(cand)
            # Simple heuristic: if we already have enough good scores, break
            if len(cands) >= self.topk and np.median([c.score_vp for c in cands]) > 0.35:
                break
        # de-dup by (movie,t0,t1,variant) keep max score_vp
        uniq = {}
        for c in cands:
            key = (c.tw.movie_id, round(c.tw.t0,3), round(c.tw.t1,3), c.variant.value)
            if key not in uniq or c.score_vp > uniq[key].score_vp:
                uniq[key] = c
        return sorted(list(uniq.values()), key=lambda x: -x.score_vp)[: self.topk]
