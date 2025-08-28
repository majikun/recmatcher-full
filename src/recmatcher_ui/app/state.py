from __future__ import annotations
from typing import Optional, Dict, Any

class InMemoryState:
    def __init__(self):
        self.project_root: Optional[str] = None
        self.paths: Dict[str,str] = {}
        self.match_segments: list[dict] = []
        self.scene_out: dict = {}
        self.explain_path: Optional[str] = None
        self.applied_changes: dict[int, dict] = {}
        self._explain_offsets: list[int] = []

    def load_project(self, root: str, movie: str | None, clip: str | None):
        from pathlib import Path; import json
        self.project_root = root
        self.paths["movie"] = movie or ""
        self.paths["clip"] = clip or ""
        def _p(*names):
            for n in names:
                p = Path(root) / n
                if p.exists(): return str(p)
            return None
        ms = _p("match_segments.json","out/match_segments.json")
        so = _p("scene_out.json","out/scene_out.json")
        ex = _p("match_explain.jsonl","out/match_explain.jsonl")
        self.match_segments = json.load(open(ms,"r",encoding="utf-8")) if ms else []
        self.scene_out = json.load(open(so,"r",encoding="utf-8")) if so else {}
        self.explain_path = ex
        self.applied_changes.clear()
        self._explain_offsets.clear()

    def build_explain_offsets(self):
        from pathlib import Path
        self._explain_offsets = []
        if not self.explain_path or not Path(self.explain_path).exists(): return 0
        off=[0]; tot=0
        with open(self.explain_path,"rb") as f:
            while True:
                line=f.readline()
                if not line: break
                tot += len(line); off.append(tot)
        self._explain_offsets = off
        return len(off)-1

    def read_explain(self, seg_id: int) -> dict | None:
        """Read explain record for given seg_id. 
        Note: seg_id and line numbers are not 1:1, so we search through the file.
        """
        from pathlib import Path
        if not self.explain_path or not Path(self.explain_path).exists():
            return None
        
        try:
            import json
            with open(self.explain_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    if data.get('type') == 'segment' and data.get('seg_id') == seg_id:
                        return data
                        
            return None
        except Exception:
            return None

STATE = InMemoryState()
