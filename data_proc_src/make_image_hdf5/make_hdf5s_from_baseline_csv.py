#!/usr/bin/env python

import argparse
import os
import subprocess
import pandas as pd
from pathlib import Path
import shutil
import sys
from glob import glob

def slide_exists(slide_id):
    tube_id = slide_id[:5]
    mnt_dirs = [os.path.join('/mnt', d) for d in os.listdir('/mnt') if os.path.isdir(os.path.join('/mnt', d))]
    for base in mnt_dirs:
        pattern = os.path.join(
            base,
            "Oncoscope",
            f"tubeID_{tube_id}",
            "*",
            f"slideID_{slide_id}",
            "bzScanner",
            "proc"
        )
        if glob(pattern):
            return True
    return False

def sanitize_type_name(type_name: str) -> str:
    """Make type names filesystem-safe."""
    return (
        type_name.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

def run_extract(
    extract_script,
    csv_path,
    output_h5,
    width,
    channels,
    starts,
    mask_dir,
    verbose
):
    cmd = [
        sys.executable,
        extract_script,
        "--data", str(csv_path),
        "--output", str(output_h5),
        "--width", str(width),
        "--channels", *channels,
        "--starts", *map(str, starts),
    ]

    if mask_dir is not None:
        cmd += ["--mask", str(mask_dir)]

    if verbose:
        cmd += ["-" + "v" * verbose]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_mask_generation(mask_script, h5_path):
    cmd = [
        sys.executable,
        mask_script,
        str(h5_path)
    ]
    print("Generating masks:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-type HDF5 files from annotated baseline CSV"
    )

    parser.add_argument("--csv", required=True, help="Annotated baseline CSV")
    parser.add_argument("--out-root", required=True, help="Root output directory")
    parser.add_argument("--extract-script", required=True, help="extract_event_images.py")
    parser.add_argument("--mask-script", default=None, help="generate_masks.py")
    parser.add_argument("--generate-masks", action="store_true", help="Run Cellpose masks")

    parser.add_argument("--width", type=int, default=75)
    parser.add_argument("--channels", nargs="+", default=["DAPI", "TRITC", "CY5", "FITC"])
    parser.add_argument("--starts", nargs="+", type=int, default=[1, 2305, 4609, 9217])
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # normalize slide_id by stripping suffix after underscore
    df["slide_id"] = (
        df["slide_id"]
        .astype(str)
        .str.split("_", n=1)
        .str[0]
    )

    print(df["type"].value_counts())

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_root / "_tmp_csvs"
    tmp_dir.mkdir(exist_ok=True)

    for type_name, type_df in df.groupby("type"):
        safe_type = sanitize_type_name(type_name)
        print(f"\n=== Processing type: {type_name} ({len(type_df)}) ===")

        type_dir = out_root / safe_type
        type_dir.mkdir(exist_ok=True)

        tmp_csv = tmp_dir / f"{safe_type}.csv"
        type_df = type_df.copy()

        if "Unnamed: 0" in type_df.columns:
            type_df = type_df.drop(columns=["Unnamed: 0"])

        type_df.columns = type_df.columns.str.strip()

        required_cols = {"slide_id", "frame_id", "x", "y"}
        missing = required_cols - set(type_df.columns)
        if missing:
            raise RuntimeError(f"Missing required columns in temp CSV: {missing}")

        type_df.to_csv(tmp_csv, index=False)

        output_h5 = type_dir / f"{safe_type}.hdf5"

        run_extract(
            extract_script=args.extract_script,
            csv_path=tmp_csv,
            output_h5=output_h5,
            width=args.width,
            channels=args.channels,
            starts=args.starts,
            mask_dir=args.mask_dir,
            verbose=args.verbose,
        )

        if args.generate_masks:
            if args.mask_script is None:
                raise ValueError("--generate-masks requires --mask-script")
            run_mask_generation(args.mask_script, output_h5)

    shutil.rmtree(tmp_dir)
    print("\nAll types processed successfully.")

if __name__ == "__main__":
    main()
