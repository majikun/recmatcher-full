## Usage

Baseline (unchanged):
  python -m recmatcher.cli.match_with_queries ...

Enable scene-level post-processing:
  python -m recmatcher.cli.match_with_queries \    --scene_enable --scene_dynamic_k --scene_vote_top_n 3 \    --scene_continuity_eps 0.08 --scene_continuity_max_gap 2 \    --scene_chain_max_skip 1 --scene_chain_len_w 0.3 --scene_chain_jump_penalty 0.2 \    --fill_enable --fill_max_gap 2 --fill_penalty 0.05 \    --scene_out /path/to/scene_out.json \    [other args...]
