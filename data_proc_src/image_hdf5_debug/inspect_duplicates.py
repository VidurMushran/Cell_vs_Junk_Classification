#!/usr/bin/env python3
import argparse
from pathlib import Path
import h5py
import pandas as pd
import hashlib
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha1_of_array(a):
    m = hashlib.sha1()
    m.update(a.tobytes())
    return m.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5", help="hdf5 path")
    p.add_argument("--max-groups", type=int, default=5, help="how many duplicate groups to inspect")
    p.add_argument("--out-dir", default="dup_inspect", help="where to save pngs")
    args = p.parse_args()

    h5path = Path(args.h5)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open file once and keep it open while we both compute hashes and save images
    with h5py.File(h5path, "r") as f:
        imgs = f["images"]
        n = imgs.shape[0]
        print("Total images:", n)

        hash_to_indices = defaultdict(list)
        for i in range(n):
            h = sha1_of_array(imgs[i])
            hash_to_indices[h].append(i)

        unique = len(hash_to_indices)
        dup_groups = [v for v in hash_to_indices.values() if len(v) > 1]
        print("Duplicate groups:", len(dup_groups))

        # load features once
        try:
            df = pd.read_hdf(h5path, key="features")
        except Exception:
            df = pd.DataFrame()

        # Inspect the first few duplicate groups
        for g_idx, indices in enumerate(dup_groups[: args.max_groups]):
            print("\n=== Duplicate Group", g_idx, "===")
            print("Indices:", indices)

            if not df.empty:
                sub = df.iloc[indices]
                cols = [c for c in ["slide_id", "frame_id", "cell_id", "cell_x", "cell_y"] if c in df.columns]
                print(sub[cols])

            # Save each image in the group so you can visually inspect
            for index in indices:
                img = imgs[index]  # safe: file open
                # show first channel as grayscale thumbnail
                fig = plt.figure(figsize=(3, 3))
                plt.imshow(img[:, :, 0], cmap="gray", interpolation="nearest")
                plt.title(f"idx={index}")
                plt.axis("off")
                out_png = out_dir / f"{h5path.stem}_idx_{index}.png"
                plt.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print("Saved:", out_png)

    print("Done. Inspect PNGs in", out_dir)


if __name__ == "__main__":
    main()
