#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import h5py
import hashlib
from collections import defaultdict

def sha1_of_array(a):
    import hashlib
    m = hashlib.sha1()
    m.update(a.tobytes())
    return m.hexdigest()

def validate(h5path):
    h5 = Path(h5path)
    manifest = h5.with_suffix(".manifest.parquet")
    print("=== VALIDATING", h5.name, "===")

    if not h5.exists():
        print("hdf5 missing:", h5)
        return

    if not manifest.exists():
        print("[WARN] manifest missing:", manifest)
        # still we can inspect images
        manifest_df = None
    else:
        manifest_df = pd.read_parquet(manifest)
        print("Manifest rows:", len(manifest_df))

    with h5py.File(h5, "r") as f:
        n_images = f["images"].shape[0]
        print("Images:", n_images)

        # hash all images (can be memory/time heavy)
        hash_to_indices = defaultdict(list)
        for i in range(n_images):
            h = sha1_of_array(f["images"][i])
            hash_to_indices[h].append(i)
    unique_images = len(hash_to_indices)
    dup_total = sum(len(v) for v in hash_to_indices.values() if len(v) > 1)
    print("Unique images:", unique_images)
    print("Total duplicate indices:", dup_total)
    if dup_total > 0:
        print("Duplicate groups:", len([1 for v in hash_to_indices.values() if len(v) > 1]))

    # compare manifest indices if present
    if manifest_df is not None:
        included = manifest_df[manifest_df.status == "included"].copy()
        idxs = included.hdf5_index.dropna().astype(int).unique()
        n_manifest = len(included)
        n_idx_unique = len(idxs)
        print("Included rows (manifest):", n_manifest)
        print("Unique hdf5_index values referenced:", n_idx_unique)

        missing_indices = set(range(n_images)) - set(idxs)
        orphan_indices = set(idxs) - set(range(n_images))
        print("Missing indices (present in images but NOT referenced by manifest):", len(missing_indices))
        if len(missing_indices) > 0 and len(missing_indices) < 50:
            print("Sample missing indices:", sorted(list(missing_indices))[:20])
        if orphan_indices:
            print("[ERROR] Manifest references indices not present in images:", sorted(list(orphan_indices))[:20])

        # sanity suggestion
        if len(missing_indices) > 0:
            print("\nRECOMMENDATION: Missing indices present. Best action: regenerate this shard (safe) OR inspect duplicate groups to see if manifest covered only unique images.")
    print("=== DONE ===\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5", nargs="+", help="hdf5 files to validate")
    args = p.parse_args()
    for h in args.h5:
        validate(h)

if __name__ == "__main__":
    main()
