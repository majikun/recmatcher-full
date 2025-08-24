from __future__ import annotations
from pathlib import Path
import yaml, json
from typing import Any, Dict

def load_yaml(path: str|Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str|Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: str|Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str|Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
