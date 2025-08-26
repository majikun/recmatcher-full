from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import cv2, numpy as np
from tqdm import tqdm
from ..types import TimeWindow, CropVariant, EmbeddingRecord
from ..utils.io import read_json, ensure_dir, write_json
from .frame_sampler import FrameSampler
from .preprocess import to_square_letterbox, to_square_centercrop, to_square_anchor
from .vpr_embed import VPRemb
from .emb_store import EmbStore
from .index_builder import IndexBuilder

def _sample_frames(cap: cv2.VideoCapture, t0: float, t1: float, n: int) -> List[np.ndarray]:
    ts = np.linspace(t0, t1, max(2, n))
    frames = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000.0)
        ok, f = cap.read()
        if not ok: 
            continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    return frames

class MovieIndexer:
    def __init__(self, out_root: str|Path):
        self.out_root = Path(out_root)

    def _encode_variant(self, movie_id: str, cap, tws: List[TimeWindow], variant: CropVariant,
                        size: int, n_frames: int, stride_s: float, vpr: VPRemb, store: EmbStore):
        batch_imgs = []
        batch_meta = []
        for tw in tqdm(tws, desc=f"encode-{variant.value}"):
            frames = _sample_frames(cap, tw.t0, tw.t1, n_frames)
            if len(frames) < max(2, n_frames//2): 
                continue
            proc_frames = []
            bbox = (0,0,1,1)
            for f in frames:
                if variant == CropVariant.LETTERBOX:
                    img, bbox = to_square_letterbox(f, size)
                elif variant == CropVariant.CENTERCROP:
                    img, bbox = to_square_centercrop(f, size)
                elif variant == CropVariant.H_LEFT:
                    img, bbox = to_square_anchor(f, size, "left")
                elif variant == CropVariant.H_RIGHT:
                    img, bbox = to_square_anchor(f, size, "right")
                else:
                    img, bbox = to_square_anchor(f, size, "center")
                proc_frames.append(img.astype(np.float32)/255.0)
            clip = np.stack(proc_frames, axis=0)
            batch_imgs.append(clip)
            batch_meta.append((tw, bbox))
            # simple batch size control
            if len(batch_imgs) >= 16:
                vecs = vpr.encode_batch(batch_imgs)
                recs = []
                for (tw0,b0), v in zip(batch_meta, vecs):
                    recs.append(EmbeddingRecord(tw0, variant, b0, size, len(proc_frames), stride_s, v.astype(np.float16)))
                store.append(variant.value, recs)
                batch_imgs.clear(); batch_meta.clear()
        if batch_imgs:
            vecs = vpr.encode_batch(batch_imgs)
            recs = []
            for (tw0,b0), v in zip(batch_meta, vecs):
                recs.append(EmbeddingRecord(tw0, variant, b0, size, len(proc_frames), stride_s, v.astype(np.float16)))
            store.append(variant.value, recs)

    def build_base(self, movie_path: str, segs_path: str, size:int=288, win_len:float=2.0, stride:float=2.0) -> None:
        movie_path = Path(movie_path); segs = read_json(segs_path)
        movie_id = movie_path.stem
        out_dir = ensure_dir(self.out_root / "movie")
        write_json(out_dir / "meta" / "segs.json", segs)
        cap = cv2.VideoCapture(str(movie_path))
        tws = FrameSampler(win_len, stride).make_windows(movie_id, segs)
        vpr = VPRemb()
        store = EmbStore(out_dir)
        # base variants
        for var in (CropVariant.LETTERBOX, CropVariant.CENTERCROP):
            self._encode_variant(movie_id, cap, tws, var, size, n_frames=12, stride_s=stride, vpr=vpr, store=store)
        store.finalize()
        # build indices
        ib = IndexBuilder(out_dir / "emb", out_dir / "index")
        for var in ("letterbox","centercrop"):
            ib.build(var)

    def build_horizontal_anchors(self, movie_path: str, segs_path: str, size:int=144, win_len:float=2.0, stride:float=1.0,
                                 only_time_ranges: Optional[List[Tuple[float,float]]] = None) -> None:
        movie_path = Path(movie_path); segs = read_json(segs_path)
        movie_id = movie_path.stem
        out_dir = ensure_dir(self.out_root / "movie")
        cap = cv2.VideoCapture(str(movie_path))
        tws_all = FrameSampler(win_len, stride).make_windows(movie_id, segs)
        if only_time_ranges:
            sel = []
            for (a,b) in only_time_ranges:
                for tw in tws_all:
                    if tw.t1 >= a and tw.t0 <= b:
                        sel.append(tw)
            tws = sel
        else:
            tws = tws_all
        vpr = VPRemb()
        store = EmbStore(out_dir)
        for var in (CropVariant.H_LEFT, CropVariant.H_CENTER, CropVariant.H_RIGHT):
            self._encode_variant(movie_id, cap, tws, var, size, n_frames=8, stride_s=stride, vpr=vpr, store=store)
        store.finalize()
        from .index_builder import IndexBuilder
        ib = IndexBuilder(out_dir / "emb", out_dir / "index")
        for var in ("h_left","h_center","h_right"):
            ib.build(var)
