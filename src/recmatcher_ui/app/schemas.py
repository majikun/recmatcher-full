from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict, List

class OpenProjectReq(BaseModel):
    root: str
    movie_path: str | None = None
    clip_path: str | None = None

class ApplyChange(BaseModel):
    seg_id: int
    chosen: Dict[str, Any]

class ApplyBatchReq(BaseModel):
    changes: List[ApplyChange]

class SaveReq(BaseModel):
    out_path: str | None = None
