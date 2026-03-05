#!/usr/bin/env python3
import argparse
import os
import re
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import h5py

# ----------------------------- Logging ---------------------------------
logger = logging.getLogger("split_by_label")
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(h)
logger.setLevel(logging.INFO)

# --------------------------- Utilities ---------------------------------
UNLABELED_PATTERNS = {"class 0", "unlabeled", "none", "no label", "nolabel", "null", "na", "n/a"}

def is_unlabeled(name: str) -> bool:
    if name is None:
        return True
    s = str(name).strip().lower()
    return (s == "") or (s in UNLABELED_PATTERNS)

def sanitize_label_for_filename(label: str) -> str:
    """
    Make a safe filename from a label value (e.g., 'Other Junk' -> 'other_junk').
    """
    s = label.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "_", s)
    s = s.strip("._-")
    return s or "unknown"

def load_features_series_and_labels_map(h5_path: str, features_key: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the features table and return:
      - features_df (unmodified)
      - label_series (as strings; unlabeled rows become empty string)
    Accepts either:
      * a 'labels' column with strings, OR
      * a numeric 'label' column and an HDF5 dataset '/labels' containing the name map.
    """
    # Read the features with pandas (HDFStore)
    try:
        features_df = pd.read_hdf(h5_path, key=features_key)
    except (KeyError, OSError) as e:
        raise RuntimeError(f"Could not read features key '{features_key}' from {h5_path}: {e}")

    # Prefer a string 'labels' column if present
    if "labels" in features_df.columns:
        s = features_df["labels"].astype("string").fillna("").str.strip()
        return features_df, s

    # Fall back to numeric 'label' column + name map in file
    if "label" in features_df.columns:
        label_ids = pd.to_numeric(features_df["label"], errors="coerce")
        name_map = None
        try:
            with h5py.File(h5_path, "r") as hf:
                if "labels" in hf:
                    raw = hf["labels"][:]
                    # decode bytes -> str if needed
                    name_map = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in raw]
        except Exception as e:
            logger.warning(f"[{h5_path}] Could not open name map at '/labels': {e}")

        if name_map is not None and len(name_map) > 0:
            def id_to_name(x):
                try:
                    xi = int(x)
                    return name_map[xi] if 0 <= xi < len(name_map) else ""
                except Exception:
                    return ""
            s = label_ids.map(id_to_name).astype("string").fillna("").str.strip()
        else:
            # No map available—fall back to 'label_<id>' strings (still allows per-label grouping)
            s = label_ids.map(lambda x: f"label_{int(x)}" if pd.notna(x) else "").astype("string")
        return features_df, s

    # Nothing usable found
    raise RuntimeError(
        f"[{h5_path}] No 'labels' (string) column and no 'label' (numeric) column present in '{features_key}'."
    )

def open_or_init_outfile(path: Path, expected_img_shape: Tuple[int, ...]) -> h5py.File:
    """
    Open an output HDF5 (create if needed). Ensure an extendable '/images' dataset with the expected shape.
    Returns an open h5py.File handle (caller must close).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    f = h5py.File(path, mode)

    if "images" in f:
        dset = f["images"]
        # Validate trailing dimensions (H, W, C)
        if dset.ndim != len(expected_img_shape) or dset.shape[1:] != expected_img_shape[1:]:
            f.close()
            raise RuntimeError(
                f"Shape mismatch for {path}: existing images dataset has shape {dset.shape}, "
                f"but incoming chunk has shape like {expected_img_shape}."
            )
    else:
        maxshape = (None,) + expected_img_shape[1:]
        chunks = (min(256, expected_img_shape[0]),) + expected_img_shape[1:]
        # Use compression for space efficiency
        f.create_dataset(
            "images",
            shape=(0,) + expected_img_shape[1:],
            maxshape=maxshape,
            chunks=chunks,
            dtype=expected_img_shape.dtype if hasattr(expected_img_shape, "dtype") else "uint16",
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=False,
        )
    return f

def append_images(f: h5py.File, images_chunk: np.ndarray) -> None:
    """
    Append a chunk of images to the extendable '/images' dataset.
    """
    dset = f["images"]
    n_old = dset.shape[0]
    n_new = n_old + images_chunk.shape[0]
    dset.resize((n_new,) + dset.shape[1:])
    dset[n_old:n_new, ...] = images_chunk

def append_features(out_path: Path, features_chunk: pd.DataFrame) -> None:
    """
    Append features (as a table) under key 'features'. Uses PyTables table format for fast append.
    """
    # Ensure a clean, simple RangeIndex and no unnamed junk
    f2 = features_chunk.copy()
    f2.reset_index(drop=True, inplace=True)
    with pd.HDFStore(out_path.as_posix(), mode="a") as store:
        store.append("features", f2, format="table", data_columns=True, index=False)

# ------------------------------ Main -----------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Split MM_cluster_*.hdf5 files into one HDF5 per unique, non-empty label "
                    "with both images and features included."
    )
    parser.add_argument("--input-dir",
                        default="/mnt/deepstore/Vidur/Junk Classification/data/unannotated/",
                        help="Directory containing MM_cluster_*.hdf5")
    parser.add_argument("--pattern", default="MM_cluster_*.hdf5",
                        help="Glob pattern for input files")
    parser.add_argument("--image-key", default="images",
                        help="HDF5 dataset key for images")
    parser.add_argument("--features-key", default="features",
                        help="Pandas HDF key for features")
    parser.add_argument("--out-dir",
                        default="/mnt/deepstore/Vidur/Junk Classification/data/by_label/",
                        help="Directory to write one HDF5 per label")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report counts only; do not write outputs")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # <<< ensure top-level output dir exists
    files = sorted(glob.glob(str(in_dir / args.pattern)))
    if not files:
        logger.error(f"No input files found under {in_dir} matching {args.pattern}")
        sys.exit(1)

    logger.info(f"Found {len(files)} input file(s). Scanning for labels...")

    # First pass: discover labels + counts (and shapes) so we can log what will be produced.
    label_counts: Dict[str, int] = {}
    # Keep track of first-seen image shapes for each label to validate consistency.
    label_img_shape: Dict[str, Tuple[int, ...]] = {}

    for h5_path in files:
        try:
            features_df, label_series = load_features_series_and_labels_map(h5_path, args.features_key)
        except Exception as e:
            logger.info(f"[scan] {h5_path}: no usable label column; skipping file ({e})")
            continue

        with h5py.File(h5_path, "r") as hf:
            if args.image_key not in hf:
                logger.warning(f"[scan] {h5_path}: missing images dataset '{args.image_key}'; skipping")
                continue
            dimg = hf[args.image_key]
            if dimg.shape[0] != len(features_df):
                logger.warning(f"[scan] {h5_path}: images N!=features N; skipping")
                continue
            file_img_shape = dimg.shape

        # keep only labeled & not-unlabeled
        used = label_series[label_series.notna()].astype(str).str.strip()
        used = used[~used.map(is_unlabeled)]
        for val, cnt in used.value_counts().items():
            label_counts[val] = label_counts.get(val, 0) + int(cnt)
            if val not in label_img_shape:
                label_img_shape[val] = file_img_shape

    if not label_counts:
        logger.warning("No non-empty labels found in any file. Nothing to do.")
        return

    logger.info("Planned outputs (label -> row count):")
    for lab, cnt in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
        logger.info(f"  {lab!r}: {cnt} rows")

    if args.dry_run:
        logger.info("Dry-run: exiting without writing.")
        return

    # Create/open output files lazily on first write per label.
    # For images we keep an open handle briefly per chunk (open/append/close) to avoid too many FDs.
    for h5_path in files:
        logger.info(f"Processing {h5_path} ...")
        try:
            features_df, label_series = load_features_series_and_labels_map(h5_path, args.features_key)
        except Exception as e:
            logger.info(f"[write] {h5_path}: no usable label column; skipping file ({e})")
            continue

        with h5py.File(h5_path, "r") as hf_in:
            if args.image_key not in hf_in:
                logger.warning(f"[write] {h5_path}: missing images; skipping")
                continue
            dimg = hf_in[args.image_key]
            if dimg.shape[0] != len(features_df):
                logger.warning(f"[write] {h5_path}: images N!=features N; skipping")
                continue

            labels_all = label_series.astype("string").fillna("").str.strip()
            mask_keep = labels_all.map(lambda s: (s != "") and (not is_unlabeled(s)))
            if not mask_keep.any():
                continue

            for lab in labels_all[mask_keep].unique().tolist():
                lab_mask = labels_all == lab
                idx = np.flatnonzero(lab_mask.values if isinstance(lab_mask, pd.Series) else lab_mask)
                if idx.size == 0:
                    continue

                feat_chunk = features_df.iloc[idx].copy()
                feat_chunk.reset_index(drop=True, inplace=True)
                feat_chunk["__source_file__"] = os.path.basename(h5_path)

                idx_sorted = np.sort(idx)
                img_chunk = dimg[idx_sorted, ...]

                out_name = sanitize_label_for_filename(str(lab))
                out_path = out_dir / f"{out_name}.hdf5"
                out_path.parent.mkdir(parents=True, exist_ok=True)   # <<< ensure parent exists

                with h5py.File(out_path, "a") as hf_out:
                    if "images" not in hf_out:
                        maxshape = (None,) + img_chunk.shape[1:]
                        chunks = (min(256, img_chunk.shape[0]),) + img_chunk.shape[1:]
                        hf_out.create_dataset(
                            "images",
                            shape=(0,) + img_chunk.shape[1:],
                            maxshape=maxshape,
                            chunks=chunks,
                            dtype=img_chunk.dtype,
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                        )
                    else:
                        dset = hf_out["images"]
                        if dset.shape[1:] != img_chunk.shape[1:]:
                            raise RuntimeError(
                                f"Image shape mismatch for label '{lab}' in {out_path}: "
                                f"existing {dset.shape[1:]} vs incoming {img_chunk.shape[1:]}"
                            )
                    # append images
                    dset = hf_out["images"]
                    n_old = dset.shape[0]
                    dset.resize((n_old + img_chunk.shape[0],) + dset.shape[1:])
                    dset[n_old:n_old + img_chunk.shape[0], ...] = img_chunk

                # append features
                with pd.HDFStore(out_path.as_posix(), mode="a") as store:
                    store.append("features", feat_chunk, format="table", data_columns=True, index=False)

    logger.info(f"Done. Wrote {len(label_counts)} HDF5 file(s) to {out_dir}")

if __name__ == "__main__":
    main()
