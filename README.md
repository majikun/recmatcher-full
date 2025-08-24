# recmatcher (implemented)

A minimally functional implementation of the **encoding (movie)** and **matching (clip)** pipeline we designed:
- Movie side: **Letterbox** (main) + **CenterCrop** (aux) indices; optional **L/C/R anchors**.
- Clip side: **multi-readouts** (tight/context + left/center/right + mirrored).
- Retrieval: two-stage (base first, anchors on demand – here simplified to always available if built).
- Re-ranking: **horizontal tile (H-Tile)** + simple dynamic energy.
- Alignment: monotonic DP across segments.

This code is designed to be runnable without DeepMind VideoPrism. If `videoprism` + `jax` are available, `VPRemb` will use them; otherwise it falls back to a light-weight embedding to keep the pipeline testable end-to-end.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Build base indices (per movie)

```bash
recmatcher-index --movie /path/MOVIE.mp4 --segs /path/segs.json --out ./store --base
```

This creates:
```
store/{movie_id}/
  emb/letterbox.npy + letterbox_id_map.csv
  emb/centercrop.npy + centercrop_id_map.csv
  index/letterbox.faiss
  index/centercrop.faiss
  meta/segs.json
```

## (Optional) Build horizontal anchors for specific ranges

```bash
recmatcher-index --movie /path/MOVIE.mp4 --segs /path/segs.json --out ./store --anchors --ranges 600-1200,2200-2600
```

## Match a clip

```bash
recmatcher-match --clip /path/clip.mp4 --clip_segs /path/clip_segs.json --store ./store --out match.json
```

Outputs per-segment alignment with fused score & breakdown.

> **Note**: This implementation focuses on architecture & interop. For production accuracy:
> - Replace the lightweight embeddings with VideoPrism (already supported by `VPRemb` if installed).
> - Enrich re-ranking with VP-based tile embeddings, add cut detectors and proper time priors.
> - Tighten the aligner with scene constraints and slope limits.
