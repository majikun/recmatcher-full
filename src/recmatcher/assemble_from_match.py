#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assemble a target short video from recmatcher match.json by cutting
time ranges from the original movie(s) at original resolution.

- Requires ffmpeg on PATH.
- Works with current match.json schema produced by recmatcher-match:
  {"segments": [{"clip_seg_id": int, "movie_id": str, "t0": float, "t1": float, ...}, ...]}

Usage:
  python assemble_from_match.py --match MATCH.json \
    --movie-template "/path/movie_assets/{movie_id}/movie.mp4" \
    --out /path/output/assembled.mp4 [--no-audio] [--fast-copy] [--tmp-dir /tmp/work]
"""

import argparse, json, os, sys, subprocess, shlex, tempfile, pathlib

def run(cmd):
    print(">>", cmd)
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

def build_ffmpeg_cut(src, t0, t1, dst, reencode=True, keep_audio=True):
    """
    Make a precise cut [t0, t1] (seconds). If reencode=True (default), cut is accurate.
    If reencode=False (fast copy), cut may snap to nearest keyframe.
    """
    t0 = max(0.0, float(t0))
    assert float(t1) > float(t0), f"Invalid range: {t0}..{t1}"
    dur = float(t1) - float(t0)

    # Accurate seek: put -ss AFTER -i when re-encoding
    if reencode:
        v_map = "-map 0:v:0"
        a_map = "-map 0:a? " if keep_audio else ""
        a_codec = "-c:a aac" if keep_audio else "-an"
        cmd = (
            f'ffmpeg -y -i {shlex.quote(src)} -ss {t0:.3f} -t {dur:.3f} '
            f'{v_map} {a_map}'
            f'-c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p '
            f'{a_codec} -movflags +faststart '
            f'{shlex.quote(dst)}'
        )
    else:
        # Fast but not always accurate: -ss BEFORE -i, stream copy
        a_copy = "-c:a copy" if keep_audio else "-an"
        cmd = (
            f'ffmpeg -y -ss {t0:.3f} -i {shlex.quote(src)} -t {dur:.3f} '
            f'-c:v copy {a_copy} '
            f'{shlex.quote(dst)}'
        )
    run(cmd)

def concat_videos(inputs, out_path, reencode_final=True, keep_audio=True):
    """
    Concat multiple mp4 segments.
    - If reencode_final=True: robust across different sources/codecs.
    - Else: try concat demuxer + copy (requires same codec/params).
    """
    if reencode_final:
        # Concatenate via intermediate list and re-encode (robust).
        # Use concat demuxer to read, then transcode to single stream.
        list_path = out_path + ".txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for p in inputs:
                f.write(f"file '{p}'\n")
        a_codec = "-c:a aac" if keep_audio else "-an"
        cmd = (
            f'ffmpeg -y -f concat -safe 0 -i {shlex.quote(list_path)} '
            f'-c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p '
            f'{a_codec} -movflags +faststart '
            f'{shlex.quote(out_path)}'
        )
        run(cmd)
        os.remove(list_path)
    else:
        # Fast copy path (requires identical codec/params across all inputs)
        list_path = out_path + ".txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for p in inputs:
                f.write(f"file '{p}'\n")
        a_copy = "-c:a copy" if keep_audio else "-an"
        cmd = (
            f'ffmpeg -y -f concat -safe 0 -i {shlex.quote(list_path)} '
            f'-c:v copy {a_copy} '
            f'{shlex.quote(out_path)}'
        )
        run(cmd)
        os.remove(list_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True, help="match.json from recmatcher-match")
    ap.add_argument("--movie-template", required=True,
                    help="Path template to movie files, e.g. '/.../movie_assets/{movie_id}/movie.mp4'")
    ap.add_argument("--out", required=True, help="output mp4")
    ap.add_argument("--tmp-dir", default=None, help="workdir for temp clips")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--fast-copy", action="store_true",
                    help="use keyframe-aligned fast cuts and concat with -c copy (may be inaccurate)")
    args = ap.parse_args()

    with open(args.match, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", data)  # fallback if it's already a list
    if not isinstance(segments, list) or len(segments) == 0:
        raise SystemExit("No segments found in match.json")

    # sort by clip segment id if present
    segments.sort(key=lambda s: int(s.get("clip_seg_id", 0)))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    workdir = args.tmp_dir or os.path.join(out_dir, ".assemble_tmp")
    os.makedirs(workdir, exist_ok=True)

    part_files = []
    for i, seg in enumerate(segments):
        mid = seg.get("movie_id")
        t0  = float(seg.get("t0", seg.get("start", 0.0)))
        t1  = float(seg.get("t1", seg.get("end", t0)))
        if t1 <= t0:
            print(f"[warn] skip invalid segment #{i}: t1<=t0 ({t0}..{t1})")
            continue
        movie_path = args.movie_template.format(movie_id=mid)
        if not os.path.exists(movie_path):
            raise SystemExit(f"Movie not found: {movie_path} (from movie_id='{mid}')")
        dst = os.path.join(workdir, f"part_{i:04d}.mp4")
        # Accurate by default (re-encode); fast path if requested
        build_ffmpeg_cut(movie_path, t0, t1, dst,
                         reencode=not args.fast_copy,
                         keep_audio=not args.no_audio)
        part_files.append(dst)

    if not part_files:
        raise SystemExit("No valid segments to assemble.")

    # Final concat; re-encode for robustness unless fast_copy specified
    concat_videos(part_files, args.out,
                  reencode_final=not args.fast_copy,
                  keep_audio=not args.no_audio)

    # Cleanup temp pieces
    for p in part_files:
        try: os.remove(p)
        except: pass
    if args.tmp_dir is None:
        try: os.rmdir(workdir)
        except: pass

    print("Done:", args.out)

if __name__ == "__main__":
    main()