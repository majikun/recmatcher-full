from __future__ import annotations
import argparse, json
from pathlib import Path
from ..utils.logging import setup_logging
from ..encoder.movie_indexer import MovieIndexer

def main():
    ap = argparse.ArgumentParser(description="Build base and/or anchor indices.")
    ap.add_argument("--movie", required=True)
    ap.add_argument("--segs", required=True, help="segs.json with scene/segment boundaries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", action="store_true", help="build base indices: letterbox + centercrop")
    ap.add_argument("--anchors", action="store_true", help="build L/C/R anchors (low-cost, on demand)")
    ap.add_argument("--ranges", default="", help="only build anchors for comma-separated ranges: e.g. 600-1200,2200-2600")
    args = ap.parse_args()

    log = setup_logging("INFO")
    mi = MovieIndexer(args.out)
    if args.base:
        log.info("Building base indices (letterbox + centercrop)")
        mi.build_base(args.movie, args.segs)
    if args.anchors:
        ranges = []
        if args.ranges.strip():
            for part in args.ranges.split(","):
                a,b = part.split("-")
                ranges.append((float(a), float(b)))
        log.info(f"Building horizontal anchors for ranges={ranges if ranges else 'ALL'}")
        mi.build_horizontal_anchors(args.movie, args.segs, only_time_ranges=ranges if ranges else None)
