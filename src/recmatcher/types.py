from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List
import numpy as np

class CropVariant(str, Enum):
    LETTERBOX = "letterbox"
    CENTERCROP = "centercrop"
    H_LEFT = "h_left"
    H_CENTER = "h_center"
    H_RIGHT = "h_right"

@dataclass(frozen=True)
class TimeWindow:
    movie_id: str
    t0: float
    t1: float
    scene_id: int = -1
    shot_id: int = -1
    near_cut_sec: float = 999.0

@dataclass
class EmbeddingRecord:
    tw: TimeWindow
    variant: CropVariant
    bbox_norm: Tuple[float,float,float,float]  # [x1,y1,x2,y2] in normalized coords after AR-preserving scale, before square op
    size: int
    n_frames: int
    stride_s: float
    vec: np.ndarray # float16 [D]

@dataclass
class Candidate:
    tw: TimeWindow
    variant: CropVariant
    score_vp: float = 0.0
    score_tile: float = 0.0
    score_dyn: float = 0.0
    score_cut: float = 0.0
    score_prior: float = 0.0
    score_fused: float = 0.0
    source_index: Optional[str] = None
    source_id: Optional[int] = None
    explain: Optional[dict] = None
