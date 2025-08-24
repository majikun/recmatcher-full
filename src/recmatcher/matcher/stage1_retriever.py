from __future__ import annotations
from typing import List, Dict
import numpy as np
import logging
from ..types import Candidate, CropVariant, TimeWindow
from ..utils.faiss_utils import FaissIndex, IdMap

class Stage1Retriever:
    def __init__(self, index_paths: Dict[CropVariant,str], id_maps: Dict[CropVariant,IdMap], topk:int=80):
        self.index_paths = index_paths
        self.id_maps = id_maps
        self.topk = topk
        self._indices = {v: (FaissIndex(p) if p else None) for v,p in index_paths.items()}

    def _search_variant(self, var, q):
        """
        旧逻辑里计算 qvecs 后直接 fi.search(...)。
        现在在调用前做日志/清洗/类型规范，并捕获异常，把错误表面化到 Python 层。
        """
        fi = self._indices[var]            # FaissIndex 封装
        qvecs = q["vecs"] if isinstance(q, dict) and "vecs" in q else q  # 兼容旧结构

        # 统一成 2D [N, D]
        qvecs = np.asarray(qvecs)
        if qvecs.ndim == 1:
            qvecs = qvecs[None, :]

        logging.getLogger(__name__).debug(
            f"[stage1] variant={var} qvecs shape={qvecs.shape} dtype={qvecs.dtype}"
        )

        # NaN/Inf 清洗 + float32
        if np.isnan(qvecs).any() or np.isinf(qvecs).any():
            logging.getLogger(__name__).warning(
                f"[stage1] variant={var} qvecs contains NaN/Inf; sanitizing"
            )
            qvecs = np.nan_to_num(qvecs, nan=0.0, posinf=1e6, neginf=-1e6)
        qvecs = qvecs.astype(np.float32, copy=False)

        # 调用 Faiss，异常时打印 variant/shape 便于定位
        try:
            D, I = fi.search(qvecs, self.topk)
        except Exception as e:
            logging.getLogger(__name__).exception(
                f"[stage1] Faiss search failed for variant={var}: {e}"
            )
            raise

        # 返回同旧版一致的三元组
        idmap = self.id_maps[var] if hasattr(self, "id_maps") else None
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
