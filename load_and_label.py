#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:10:54 2025

@author: mushran
"""

"""
Use trained CNN to label each cell in HDF5 cluster files as junk (1) or not junk (0),
writing results into the `features` dataset's 'label' column.

Example:
    python label_clusters_with_model.py \
        --model_path junk_bin_out/best_binary_model_trainfrac_90.pt \
        --cluster_dir "/mnt/deepstore/Vidur/Junk Classification/data_model_labels/unannotated" \
        --device cuda
"""

import os, glob, argparse, h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------------
# Model definition (same as training)
# -------------------------------
class SimpleCNN(torch.nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)


# -------------------------------
# Image normalization / resizing
# -------------------------------
def normalize_image(arr, ch_mean, ch_std, target_hw=75):
    H, W, C = arr.shape
    if H != target_hw or W != target_hw:
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(target_hw, target_hw),
                          mode="bilinear", align_corners=False)
        arr = t.squeeze(0).permute(1, 2, 0).numpy()
    arr = arr.astype(np.float32)
    for c in range(C):
        arr[..., c] = (arr[..., c] - ch_mean[c]) / (ch_std[c] + 1e-8)
    return arr


# -------------------------------
# Label all clusters
# -------------------------------
@torch.no_grad()
def label_clusters(model, cluster_files, device, ch_mean, ch_std, batch=256, target_hw=75):
    model.eval()
    for path in tqdm(cluster_files, desc="Labeling clusters"):
        try:
            with h5py.File(path, "r") as f:
                imgs = f["images"]
                n_cells = imgs.shape[0]
                features_df = pd.read_hdf(path, key="features")

                labels = np.zeros((n_cells,), dtype=np.uint8)
                for start in range(0, n_cells, batch):
                    end = min(n_cells, start + batch)
                    batch_imgs = []
                    for i in range(start, end):
                        arr = imgs[i].astype(np.float32)
                        arr = normalize_image(arr, ch_mean, ch_std, target_hw)
                        batch_imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
                    xb = torch.stack(batch_imgs).to(device)
                    prob1 = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
                    labels[start:end] = (prob1 >= 0.5).astype(np.uint8)
        except Exception as e:
            print(f"[WARN] Failed reading {path}: {e}")
            continue

        # write labels back *after* closing h5py file
        try:
            features_df["label"] = labels
            with pd.HDFStore(path, mode="r+") as store:
                if "features" in store:
                    del store["features"]
                store.put("features", features_df, format="table")
            print(f"[OK] Updated 'label' column in {os.path.basename(path)}")
        except Exception as e:
            print(f"[ERROR] Could not write labels to {path}: {e}")


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--cluster_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--target_hw", type=int, default=75)
    args = ap.parse_args()

    device = args.device
    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    model = SimpleCNN(in_ch=4).to(device)
    model.load_state_dict(ckpt["state_dict"])
    ch_mean = ckpt.get("mean", [0, 0, 0, 0])
    ch_std = ckpt.get("std", [1, 1, 1, 1])
    print(f"Loaded model from {args.model_path}")

    # Find HDF5 cluster files
    cluster_files = sorted(glob.glob(os.path.join(args.cluster_dir, "*.hdf5")))
    if not cluster_files:
        print(f"No HDF5 files found in {args.cluster_dir}")
        return

    # Label in place
    label_clusters(
        model,
        cluster_files,
        device,
        ch_mean=ch_mean,
        ch_std=ch_std,
        batch=args.batch_size,
        target_hw=args.target_hw,
    )

if __name__ == "__main__":
    main()