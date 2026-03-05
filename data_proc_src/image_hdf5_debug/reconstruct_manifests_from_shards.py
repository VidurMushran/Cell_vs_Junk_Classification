#!/usr/bin/env python3
import pandas as pd
import h5py
from pathlib import Path
import argparse

def reconstruct(type_dir: Path, csv_root: Path):
    type_dir = Path(type_dir)
    csv_root = Path(csv_root)

    h5s = sorted(type_dir.glob("*.hdf5"))
    h5s = [p for p in h5s if ":" in p.name]  # only shard hdf5s

    if not h5s:
        print("No shard hdf5 files found.")
        return

    for h5_path in h5s:
        job = h5_path.stem  # e.g. Not_interesting:shard_04
        csv_path = csv_root / f"{job}.csv"

        if not csv_path.exists():
            print(f"⚠️ CSV missing for {job}, skipping")
            continue

        df = pd.read_csv(csv_path, dtype={"slide_id": str})
        manifest_path = h5_path.with_suffix(".manifest.parquet")

        with h5py.File(h5_path, "r") as f:
            n_imgs = f["images"].shape[0]

        if n_imgs > len(df):
            print(f"⚠️ {job}: more images than rows? ({n_imgs} vs {len(df)})")

        records = []

        # included rows (by index)
        for i in range(min(n_imgs, len(df))):
            row = df.iloc[i]
            records.append({
                **row.to_dict(),
                "status": "included",
                "reason": "",
                "hdf5_index": i,
            })

        # remaining rows = skipped (ran out of images)
        for i in range(n_imgs, len(df)):
            row = df.iloc[i]
            records.append({
                **row.to_dict(),
                "status": "skipped_unknown",
                "reason": "no_image_written",
                "hdf5_index": None,
            })

        manifest = pd.DataFrame(records)
        manifest.to_parquet(manifest_path, engine="pyarrow")
        print(f"✔ wrote {manifest_path.name} ({len(manifest)} rows)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--type-dir", required=True)
    p.add_argument("--csv-root", required=True,
                   help="Directory containing the per-shard CSVs (_tmp_csvs)")
    args = p.parse_args()

    reconstruct(Path(args.type_dir), Path(args.csv_root))

if __name__ == "__main__":
    main()
