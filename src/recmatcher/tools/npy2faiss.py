#!/usr/bin/env python3
import argparse, os, csv, numpy as np, faiss
from pathlib import Path

VARIANTS = ["letterbox", "centercrop", "h_left", "h_center", "h_right"]

def load_id_map(csv_path):
    ids = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r, None)
        # 兼容两种格式: faiss_id,tw_json 或 仅时间窗
        for row in r:
            ids.append(row[0])
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie_dir", required=True, help=".../smart_reclip_new/movie")
    ap.add_argument("--metric", choices=["ip","l2"], default="ip", help="faiss 度量 (默认内积)")
    ap.add_argument("--normalize", action="store_true", help="建库前对向量做 L2 归一化")
    args = ap.parse_args()

    mdir = Path(args.movie_dir)
    emb_dir = mdir/"emb"
    idx_dir = mdir/"index"
    idx_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    for var in VARIANTS:
        npy = emb_dir/f"{var}.npy"
        csvp = emb_dir/f"{var}_id_map.csv"
        out = idx_dir/f"{var}.faiss"
        if not npy.exists() or not csvp.exists():
            print(f"[skip] {var}: missing {npy.name} or {csvp.name}")
            continue
        if out.exists():
            print(f"[ok]   {var}: {out.name} already exists")
            built += 1
            continue

        x = np.load(npy, mmap_mode="r").astype("float32")
        x = np.array(x)  # materialize
        if args.normalize:
            faiss.normalize_L2(x)

        d = x.shape[1]
        index = faiss.IndexFlatIP(d) if args.metric=="ip" else faiss.IndexFlatL2(d)
        index.add(x)
        faiss.write_index(index, str(out))

        # 轻量一致性检查：行数一致
        ids = load_id_map(csvp)
        if len(ids) != x.shape[0]:
            print(f"[warn] {var}: id_map rows ({len(ids)}) != embeddings ({x.shape[0]})")

        print(f"[done] {var}: built {out.name} with {x.shape[0]} vectors, dim={d}")
        built += 1

    if built == 0:
        raise SystemExit("No indices built. Check emb/*.npy and *_id_map.csv availability.")

if __name__ == "__main__":
    main()