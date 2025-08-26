# PR: Scene-level consensus + chain post-processing

- Add `recmatcher.post.scene_consensus` and `recmatcher.post.scene_chain`.
- `--scene_enable` toggles scene-level consensus and chain/fill post-processing.
- Writes extra explain lines: `type: "scene_meta"` and `type: "scene_chain"` in `match_explain.jsonl`.
- Optional `--scene_out` to dump per-clip-scene mapping summary.
- Keeps `match.json` schema unchanged (lightweight). Scene post-processing only updates `matched_orig_seg` and narrows `top_matches`.
