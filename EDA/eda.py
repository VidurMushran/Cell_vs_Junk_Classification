#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 20:24:40 2025

@author: mushran
"""

# eda_events.py  (images-only HDF5 EDA)
# - Summarizes n_cells from images dataset (N, H, W, 4)
# - Computes accurate proportions per file and per folder
# - Quick viewers for single images and grids

import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = "/mnt/deepstore/Vidur/Junk Classification/data"  # <-- EDIT ME
SUBFOLDERS = [
    "junk_annotated",
    "rare_cells_annotated",
    "wbcs_annotated",
    "unannotated",
    "common_cell"
    "Common_Cell",
    "Dapi_",
    "Dapi__Cell",
    "Interesting",
    "Not_classified",
    "Not_interesting",
    "Not_sure",
    "Rare_Cell",
    "Unpacked",
    "Unsure"
]
SAVE_SUMMARY_CSV = True
SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "eda_summary.csv")

# ----------------------------
# HDF5 probes
# ----------------------------
def probe_images(hdf_path: str) -> Tuple[bool, Optional[Tuple[int, ...]], Optional[List[str]]]:
    """
    Return (has_images, image_shape, channels) for an HDF5.
    Expects 'images' as (N, H, W, C). Channels optional.
    """
    try:
        with h5py.File(hdf_path, "r") as f:
            if "images" not in f:
                return False, None, None
            shape = tuple(int(x) for x in f["images"].shape)  # (N,H,W,C)
            chans = None
            if "channels" in f:
                raw = f["channels"][()]
                try:
                    chans = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw]
                except Exception:
                    chans = None
            return True, shape, chans
    except Exception:
        return False, None, None

# ----------------------------
# Summaries
# ----------------------------
def summarize_hdf5(hdf_path: str) -> dict:
    folder = os.path.basename(os.path.dirname(hdf_path))
    fname = os.path.basename(hdf_path)
    has_images, img_shape, channels = probe_images(hdf_path)
    n_cells = int(img_shape[0]) if (has_images and img_shape is not None and len(img_shape) >= 1) else 0
    return {
        "folder": folder,
        "filename": fname,
        "path": hdf_path,
        "has_images": has_images,
        "image_shape": img_shape,
        "channels": channels,
        "n_cells": n_cells,
    }

def crawl_and_summarize(base_dir: str, subfolders: List[str]) -> pd.DataFrame:
    records = []
    for sub in subfolders:
        subdir = os.path.join(base_dir, sub)
        files = sorted(glob.glob(os.path.join(subdir, "*.hdf5")))
        for hdf in files:
            records.append(summarize_hdf5(hdf))
    df = pd.DataFrame(records)
    if df.empty:
        print("No .hdf5 files found. Check BASE_DIR and SUBFOLDERS.")
        return df

    # proportions overall
    total_cells = int(df["n_cells"].sum())
    df["prop_of_all"] = df["n_cells"] / total_cells if total_cells > 0 else 0.0

    # per-folder proportions
    folder_totals = df.groupby("folder")["n_cells"].sum().rename("folder_total_cells")
    df = df.merge(folder_totals, on="folder", how="left")
    df["prop_within_folder"] = df["n_cells"] / df["folder_total_cells"]
    return df

def print_overview(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n=== File-level Summary ===")
    cols = ["folder", "filename", "n_cells", "prop_of_all", "has_images", "image_shape"]
    print(df[cols].to_string(index=False))

    print("\n=== Folder Totals (cells) ===")
    folder_tot = df.groupby("folder")["n_cells"].sum().sort_values(ascending=False)
    print(folder_tot.to_string())

    print("\n=== Overall total cells ===")
    print(int(df["n_cells"].sum()))

# ----------------------------
# Image viewers (images-only)
# ----------------------------
def _load_images(hdf_path: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    try:
        with h5py.File(hdf_path, "r") as f:
            if "images" not in f:
                return None, None
            imgs = f["images"][()]  # (N,H,W,C)
            chans = None
            if "channels" in f:
                raw = f["channels"][()]
                try:
                    chans = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw]
                except Exception:
                    chans = None
            return imgs, chans
    except Exception:
        return None, None

def show_event_image(hdf_path: str, idx: int = 0, title: Optional[str] = None):
    imgs, chans = _load_images(hdf_path)
    if imgs is None:
        print(f"[No images] '{os.path.basename(hdf_path)}' has no 'images' dataset.")
        return
    if idx < 0 or idx >= imgs.shape[0]:
        print(f"Index {idx} out of range [0, {imgs.shape[0]-1}]")
        return

    img = imgs[idx]  # (H,W,C)
    H, W, C = img.shape
    fig, axes = plt.subplots(1, C, figsize=(3.6*C, 3.6))
    if C == 1:
        axes = [axes]
    for c in range(C):
        axes[c].imshow(img[:, :, c], cmap="gray")
        ch_name = chans[c] if (chans and c < len(chans)) else f"Ch{c}"
        axes[c].set_title(ch_name)
        axes[c].axis("off")

    if title is None:
        title = f"{os.path.basename(hdf_path)} | idx={idx}"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_grid(hdf_path: str, indices: List[int], suptitle: Optional[str] = None):
    imgs, _ = _load_images(hdf_path)
    if imgs is None:
        print(f"[No images] '{os.path.basename(hdf_path)}' has no 'images' dataset.")
        return
    if len(indices) == 0:
        print("No indices provided.")
        return

    n = len(indices)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(indices):
        r = i // cols
        c = i % cols
        if idx < 0 or idx >= imgs.shape[0]:
            axes[r, c].set_axis_off()
            continue
        axes[r, c].imshow(imgs[idx, :, :, 0], cmap="gray")  # thumbnail: show Ch0
        axes[r, c].set_title(f"idx {idx}", fontsize=9)
        axes[r, c].axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def sample_and_show(hdf_path: str, n: int = 12, seed: int = 0):
    imgs, _ = _load_images(hdf_path)
    if imgs is None:
        print("No images available in this file.")
        return
    N = imgs.shape[0]
    n = min(n, N)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(np.arange(N), size=n, replace=False).tolist())
    show_grid(hdf_path, indices, suptitle=f"{os.path.basename(hdf_path)} | n={len(indices)}")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("=== EDA: scanning HDF5s ===")
    df_summary = crawl_and_summarize(BASE_DIR, SUBFOLDERS)
    if not df_summary.empty:
        print_overview(df_summary)
        if SAVE_SUMMARY_CSV:
            df_out = df_summary.copy()
            # pretty-print image_shape & channels for CSV
            df_out["image_shape"] = df_out["image_shape"].apply(lambda s: "" if s is None else str(tuple(s)))
            df_out["channels"] = df_out["channels"].apply(lambda c: "" if c is None else ",".join(c))
            df_out.to_csv(SUMMARY_CSV_PATH, index=False)
            print(f"\nSaved summary CSV to: {SUMMARY_CSV_PATH}")

        # Quick usage examples (run these in the Spyder console after script loads):
        # p0 = df_summary.loc[0, "path"]
        # show_event_image(p0, idx=0)     # show all channels for one event
        # sample_and_show(p0, n=12)       # random grid of Ch0 thumbnails
