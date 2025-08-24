#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Draw square-coordinate schematics for:
  1) Query (9:16) -> Tight (center-crop) vs Context (letterbox)
  2) Movie -> Letterbox (scale by long side)
  3) Movie -> Horizontal anchors L/C/R (scale by short side; 1x1 crops)

No colors are specified; uses default matplotlib styling.
Each figure is a separate canvas (no subplots).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_query_diagram(out_path: str, ar_query: float = 9/16, size_label: int = 288):
    """
    Square output is normalized to [0,1]x[0,1].
    - Context (letterbox): scale by long side (height) → width=ar_query (<1 for 9:16), pad L/R
    - Tight   (center-crop): scale by short side (width) → height=1/ar_query (>1), crop top/bottom
    """
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    # Outer square = output SxS
    ax.add_patch(Rectangle((0,0), 1, 1, fill=False, linewidth=2))
    ax.text(0.02, 0.98, f"Output square ({size_label}×{size_label})", va='top')

    # Context / Letterbox (keep full 9:16, pad L/R)
    w_ctx = float(ar_query)     # since height maps to 1
    x_ctx = (1 - w_ctx) / 2.0
    ax.add_patch(Rectangle((x_ctx, 0), w_ctx, 1, fill=False, hatch='//', linewidth=1.5))
    ax.text(0.02, 0.92, "Context (letterbox): scale by long side (height)\n→ keep full frame, pad left/right", va='top')

    # Tight / Center-crop (crop top/bottom)
    h_tight = 1.0 / float(ar_query)  # >1 for 9:16
    y_tight = (1 - h_tight) / 2.0    # negative value means extends above/below
    ax.add_patch(Rectangle((0, y_tight), 1, h_tight, fill=False, linestyle='--', linewidth=1.5))
    ax.text(0.02, 0.82, "Tight (center-crop): scale by short side (width)\n→ crop top/bottom", va='top')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def draw_movie_letterbox(out_path: str, ar_movie: float = 2.39):
    """
    Movie letterbox (scale by long side = max(H,W)):
    For a wide AR (>1), long side is width, so height becomes 1/ar_movie (<1).
    The content is a horizontal strip centered vertically (pads on top/bottom).
    """
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    ax.add_patch(Rectangle((0,0), 1, 1, fill=False, linewidth=2))
    ax.text(0.02, 0.98, "Movie: letterbox index (scale by long side)", va='top')

    h = 1.0 / float(ar_movie)
    y0 = (1 - h) / 2.0
    ax.add_patch(Rectangle((0, y0), 1, h, fill=False, hatch='///', linewidth=1.5))
    ax.text(0.02, 0.92, "Content strip (no crop)\n→ pads on top/bottom", va='top')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def draw_movie_anchors(out_path: str, ar_movie: float = 2.39, overlap: float = 0.0):
    """
    Movie horizontal anchors:
      1) Scale by short side (height) -> height=1, width=ar_movie (>1) → extends beyond square horizontally
      2) Take 1×1 crops at Left / Center / Right.
    If you want explicit overlap between anchors, set `overlap` in [0,1).
    overlap=0 → L/C/R just-touch (x positions: 0, (w-1)/2, (w-1))
    overlap>0 → step = 1 - overlap; centers move closer → more overlap
    """
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    ax.add_patch(Rectangle((0,0), 1, 1, fill=False, linewidth=2))
    ax.text(0.02, 0.98, "Movie: horizontal anchors (scale by short side)", va='top')

    # Resized image (height=1, width=ar_movie)
    w = float(ar_movie)
    x_start = (1 - w) / 2.0  # negative if w>1 (extends beyond square)
    ax.add_patch(Rectangle((x_start, 0), w, 1, fill=False, linestyle=':', linewidth=1.2))
    ax.text(0.02, 0.92, "Resized full image (H=1, W=AR)\n→ crop L/C/R 1×1 windows", va='top')

    # Compute L/C/R with optional overlap
    step = max(1e-6, 1.0 - float(overlap))  # distance between left edges of successive crops
    # Ideal left edges if we were to place three crops with given step across width w:
    # We'll pin Left at x_start, Right at x_start + (w - 1), and set Center midway.
    xL = x_start
    xR = x_start + (w - 1.0)
    xC = x_start + (w - 1.0)/2.0

    # Draw squares
    ax.add_patch(Rectangle((xL, 0), 1, 1, fill=False, linewidth=1.8))
    ax.text(xL + 0.02, 0.03, "L", va='bottom')

    ax.add_patch(Rectangle((xC, 0), 1, 1, fill=False, linewidth=1.8, linestyle='--'))
    ax.text(xC + 0.02, 0.03, "C", va='bottom')

    ax.add_patch(Rectangle((xR, 0), 1, 1, fill=False, linewidth=1.8, linestyle='-.'))
    ax.text(xR + 0.02, 0.03, "R", va='bottom')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Draw square-coordinate diagrams for query/movie windows.")
    ap.add_argument("--out", required=True, help="Output directory for PNGs")
    ap.add_argument("--ar-query", type=float, default=9/16, help="Aspect ratio of query (width/height). Default=9/16≈0.5625")
    ap.add_argument("--ar-movie", type=float, default=2.39, help="Aspect ratio of movie (width/height). Default=2.39")
    ap.add_argument("--size", type=int, default=288, help="Target square size (label only)")
    ap.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    ap.add_argument("--overlap", type=float, default=0.0, help="Anchor overlap in [0,1). Only affects visualization.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    p1 = os.path.join(args.out, "query_9x16_diagram.png")
    p2 = os.path.join(args.out, "movie_letterbox_diagram.png")
    p3 = os.path.join(args.out, "movie_anchors_diagram.png")

    # Draw with requested DPI
    plt.rcParams["figure.dpi"] = args.dpi

    draw_query_diagram(p1, ar_query=args.ar_query, size_label=args.size)
    draw_movie_letterbox(p2, ar_movie=args.ar_movie)
    draw_movie_anchors(p3, ar_movie=args.ar_movie, overlap=args.overlap)

    print("Saved:", p1)
    print("Saved:", p2)
    print("Saved:", p3)

if __name__ == "__main__":
    main()