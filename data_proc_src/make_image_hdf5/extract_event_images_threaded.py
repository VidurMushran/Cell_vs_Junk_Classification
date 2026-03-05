#!/usr/bin/env python3
import argparse
import os, glob
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from slideutils.utils.frame import Frame
from slideutils.utils import utils

MAX_FAILURES_PER_SLIDE = 10


# -------------------------
# Helpers
# -------------------------
def find_slide_directory(slide_id):
    tube_id = slide_id[:5]
    for d in os.listdir("/mnt"):
        base = os.path.join("/mnt", d)
        pattern = os.path.join(
            base, "Oncoscope", f"tubeID_{tube_id}", "*",
            f"slideID_{slide_id}", "bzScanner", "proc"
        )
        matches = glob.glob(pattern)
        if matches:
            image_dir = matches[0]
            if glob.glob(os.path.join(image_dir, "Tile*.tif")):
                return image_dir, "Tile%06d.tif"
            if glob.glob(os.path.join(image_dir, "Tile*.jpg")):
                return image_dir, "Tile%06d.jpg"
    return None, None


# -------------------------
# Core
# -------------------------
def extract(args):
    logger = utils.get_logger(__name__, args.verbose)

    df = pd.read_csv(args.data, dtype={"slide_id": str})
    df["frame_id"] = df["frame_id"].astype(int)

    manifest_path = Path(args.output).with_suffix(".manifest.parquet")

    # ---------- RESUME ----------
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
        done_keys = set(zip(manifest.slide_id, manifest.frame_id, manifest.cell_id))
        df["_key"] = list(zip(df.slide_id, df.frame_id, df.cell_id))
        df = df[~df["_key"].isin(done_keys)].drop(columns="_key")
        logger.info(f"Resuming, {len(df)} rows left")

    if df.empty:
        logger.info("Nothing left to process.")
        return

    # ---------- Slide discovery ----------
    slide_dirs = {}
    for sid in df.slide_id.unique():
        d, fmt = find_slide_directory(sid)
        if d:
            slide_dirs[sid] = (d, fmt)

    # ---------- HDF5 initialization ----------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with h5py.File(args.output, "a") as f:
        if "images" not in f:
            f.create_dataset(
                "images",
                shape=(0, args.width, args.width, len(args.channels)),
                maxshape=(None, args.width, args.width, len(args.channels)),
                dtype="uint16",
            )

        if "channels" not in f:
            f.create_dataset("channels", data=np.array(args.channels, dtype="S"))

    records = []

    # ---------- Processing ----------
    for _, row in df.iterrows():
        slide_id = row.slide_id
        frame_id = row.frame_id
        cell_id = row.cell_id

        if slide_id not in slide_dirs:
            records.append({
                **row,
                "status": "skipped_bad_slide",
                "reason": "slide_not_found",
                "hdf5_index": None
            })
            continue

        image_dir, fmt = slide_dirs[slide_id]
        paths = [f"{image_dir}/{fmt % (frame_id + j - 1)}" for j in args.starts]

        if not all(os.path.exists(p) for p in paths):
            records.append({
                **row,
                "status": "skipped_missing_image",
                "reason": "tile_missing",
                "hdf5_index": None
            })
            continue

        try:
            frame = Frame(frame_id, args.channels, paths)
            frame.readImage()
            crops, _ = frame.extract_crops(
                pd.DataFrame([row]), args.width, False
            )

            if len(crops) != 1:
                raise RuntimeError("Expected exactly one crop")

            # -------- Write image --------
            with h5py.File(args.output, "a") as f:
                imgs = f["images"]
                idx0 = imgs.shape[0]
                imgs.resize(idx0 + 1, axis=0)
                imgs[idx0] = crops[0]

            # -------- Write features row --------
            row_df = pd.DataFrame([row]).reset_index(drop=True)
            row_df["image_id"] = idx0
            row_df["__source_h5__"] = Path(args.output).name

            with pd.HDFStore(args.output, mode="a") as store:
                store.append(
                    "features",
                    row_df,
                    format="table",
                    data_columns=True,
                    index=False
                )

            records.append({
                **row,
                "status": "included",
                "reason": "",
                "hdf5_index": idx0
            })

        except Exception as e:
            records.append({
                **row,
                "status": "skipped_error",
                "reason": str(e),
                "hdf5_index": None
            })

        # periodic flush
        if len(records) >= 1000:
            pd.DataFrame(records).to_parquet(
                manifest_path, engine="pyarrow", append=True
            )
            records = []

    if records:
        pd.DataFrame(records).to_parquet(
            manifest_path, engine="pyarrow", append=True
        )

    logger.info("Finished extraction")


# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=75)
    p.add_argument("--channels", nargs="+", default=["DAPI","TRITC","CY5","FITC"])
    p.add_argument("--starts", nargs="+", type=int, default=[1,2305,4609,9217])
    p.add_argument("-v","--verbose", action="count", default=0)
    args = p.parse_args()
    extract(args)

if __name__ == "__main__":
    main()
