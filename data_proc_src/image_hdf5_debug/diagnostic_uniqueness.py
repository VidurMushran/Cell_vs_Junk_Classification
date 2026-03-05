#!/usr/bin/env python3
import argparse
from pathlib import Path
import h5py
import hashlib
from collections import defaultdict
import csv

def sha1_of_array(a):
    m = hashlib.sha1()
    m.update(a.tobytes())
    return m.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5", help="hdf5 file path")
    p.add_argument("--dump-duplicates", help="csv to write duplicate groups (index,hash,count)")
    p.add_argument("--list-sample", type=int, default=0, help="if >0, print a sample of duplicate groups")
    args = p.parse_args()

    h5 = Path(args.h5)
    with h5py.File(h5, "r") as f:
        imgs = f["images"]
        n = imgs.shape[0]
        print("Total images:", n)

        hash_to_indices = defaultdict(list)
        for i in range(n):
            h = sha1_of_array(imgs[i])
            hash_to_indices[h].append(i)

    unique = len(hash_to_indices)
    dup_counts = [len(v) for v in hash_to_indices.values() if len(v) > 1]
    n_dup = sum(dup_counts)
    print("Unique images:", unique)
    print("Duplicate image indices (total duplicates):", n_dup)
    if dup_counts:
        print("Duplicate groups:", len(dup_counts), "largest group size:", max(dup_counts))

    if args.dump_duplicates:
        with open(args.dump_duplicates, "w", newline="") as outf:
            w = csv.writer(outf)
            w.writerow(["hash","count","indices"])
            for h, idxs in hash_to_indices.items():
                if len(idxs) > 1:
                    w.writerow([h, len(idxs), ";".join(map(str, idxs))])
        print("Wrote duplicate groups to", args.dump_duplicates)

    if args.list_sample and dup_counts:
        print("Sample duplicate groups:")
        c = 0
        for h, idxs in hash_to_indices.items():
            if len(idxs) > 1:
                print(h, len(idxs), idxs[:10])
                c += 1
                if c >= args.list_sample:
                    break

if __name__ == "__main__":
    main()
