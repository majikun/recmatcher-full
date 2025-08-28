from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    redis_url: str | None = os.environ.get("REDIS_URL")
    allowed_origin: str = os.environ.get("ALLOWED_ORIGIN", "*")

settings = Settings()
