#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 21:07:25 2025

@author: mushran
"""

# train_and_test_hdf5_cnn.py
# Trains (1) a 4-ch CNN from scratch and (2) a 4-ch ResNet18 (finetuned) on annotated HDF5s,
# then predicts on unannotated cluster HDF5s and prints class proportions per cluster.

import os
import math
import time
import glob
import h5py
import json
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models

# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish (may slow a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_h5s(root, subfolder):
    d = os.path.join(root, subfolder)
    return sorted(glob.glob(os.path.join(d, "*.hdf5")))

def pretty_time(s):
    if s < 60: return f"{s:.1f}s"
    m = int(s // 60); sec = s - 60*m
    return f"{m}m{sec:.0f}s"

# ----------------------------
# HDF5 dataset that streams rows
# ----------------------------
class H5ImageDataset(Dataset):
    """
    Streams 4-channel images from multiple HDF5s.
    Expects dataset 'images' shaped (N, H, W, C). No labels inside HDF5;
    we assign labels by the folder they came from.
    - Resize to (75, 75) if needed.
    - Per-channel z-score normalization using provided mean/std.
    """
    def __init__(self, items, label_map, channel_mean=None, channel_std=None, target_hw=75, dtype=torch.float32):
        """
        items: list of dicts with keys:
           {"path": <h5 path>, "label": <int>, "rows": [idx0, idx1, ...]}
        label_map: {int->str} for reference (not used here)
        channel_mean/std: list/np.array length 4 (computed on train)
        """
        self.items = []
        for it in items:
            # store path/label and rows; open file lazily each __getitem__
            for ridx in it["rows"]:
                self.items.append((it["path"], it["label"], ridx))
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.target_hw = target_hw
        self.dtype = dtype

        # cache: path -> (h5file, dset)
        self._handles = {}

    def __len__(self):
        return len(self.items)

    def _open(self, path):
        if path in self._handles:
            return self._handles[path]
        f = h5py.File(path, "r")
        d = f["images"]
        self._handles[path] = (f, d)
        return self._handles[path]

    def _resize_np(self, img, target_hw=75):
        # img: (H, W, C) uint16 or uint8 -> bilinear resize using torch for consistency
        H, W, C = img.shape
        if H == target_hw and W == target_hw:
            return img
        # to torch, add N,C,H,W
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()  # 1,C,H,W
        t = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
        out = t.squeeze(0).permute(1,2,0).numpy()
        return out

    def _normalize(self, x):
        # x float32 in [0, ~65535] (if uint16 input). Normalize per-channel with provided stats.
        # If stats are None, do simple scaling to [0,1] by 99.5 pct per channel to be robust.
        if self.channel_mean is not None and self.channel_std is not None:
            for c in range(x.shape[-1]):
                x[..., c] = (x[..., c] - self.channel_mean[c]) / (self.channel_std[c] + 1e-8)
            return x
        else:
            # percentile scaling
            x2 = x.copy()
            for c in range(x.shape[-1]):
                p = np.percentile(x[..., c], 99.5)
                p = max(p, 1.0)
                x2[..., c] = x[..., c] / p
            x2 = np.clip(x2, 0.0, 1.0)
            return x2

    def __getitem__(self, idx):
        path, label, ridx = self.items[idx]
        f, d = self._open(path)
        arr = d[ridx]  # (H,W,C)
        # to float32
        arr = arr.astype(np.float32)
        # resize if needed
        arr = self._resize_np(arr, target_hw=self.target_hw)
        # normalize
        arr = self._normalize(arr)
        # to CHW tensor
        x = torch.from_numpy(arr).permute(2,0,1).to(self.dtype)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    def close(self):
        for p,(f,_) in self._handles.items():
            try: f.close()
            except: pass
        self._handles = {}

# ----------------------------
# Compute channel mean/std on a subset
# ----------------------------
def estimate_channel_stats(train_items, max_samples=5000, target_hw=75):
    # sample up to max_samples across all training items
    rng = np.random.default_rng(0)
    flat = []
    for it in train_items:
        rows = it["rows"]
        if len(rows) == 0: continue
        take = min(len(rows), max(1, max_samples // max(1, len(train_items))))
        chosen = rng.choice(rows, size=take, replace=False)
        flat.append({"path": it["path"], "rows": chosen, "label": it["label"]})
    # stream and accumulate
    sums = None
    sums2 = None
    count = 0
    # temp dataset to reuse resize/robust scaling disabled here (we want raw to compute global stats)
    for it in flat:
        with h5py.File(it["path"], "r") as f:
            d = f["images"]
            for ridx in it["rows"]:
                arr = d[ridx].astype(np.float32)  # HWC
                # resize
                H,W,C = arr.shape
                if H != target_hw or W != target_hw:
                    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
                    t = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
                    arr = t.squeeze(0).permute(1,2,0).numpy()
                # accumulate per-channel
                if sums is None:
                    sums = np.zeros((C,), dtype=np.float64)
                    sums2 = np.zeros((C,), dtype=np.float64)
                # mean over spatial dims
                ch_means = arr.reshape(-1, C).mean(axis=0)
                ch_vars = arr.reshape(-1, C).var(axis=0)
                sums += ch_means
                sums2 += ch_vars + ch_means**2  # E[X^2] = Var + mean^2
                count += 1
    if count == 0:
        return None, None
    mean = (sums / count).astype(np.float32)
    ex2 = (sums2 / count).astype(np.float32)
    std = np.sqrt(np.maximum(ex2 - mean**2, 1e-8)).astype(np.float32)
    return mean.tolist(), std.tolist()

# ----------------------------
# Models
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 75 -> 37

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 37 -> 18

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 18 -> 9
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return self.head(x)

def make_resnet18_4ch(n_classes=3, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    # adapt first conv to 4-ch
    w = m.conv1.weight  # (64,3,7,7)
    new_w = torch.zeros((w.shape[0], 4, w.shape[2], w.shape[3]))
    new_w[:, :3, :, :] = w
    # init 4th channel as mean of RGB kernels
    new_w[:, 3:4, :, :] = w.mean(dim=1, keepdim=True)
    m.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight = nn.Parameter(new_w)
    # replace classifier
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m

# ----------------------------
# Training / Eval
# ----------------------------
def compute_class_weights(items, n_classes):
    counts = np.zeros((n_classes,), dtype=np.int64)
    for it in items:
        counts[it["label"]] += len(it["rows"])
    weights = counts.sum() / np.maximum(counts, 1)
    return torch.tensor(weights, dtype=torch.float32)

def train_one(model, loader, optimizer, device, criterion, use_amp=False):
    model.train()
    tot = 0.0; n = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot += loss.item() * xb.size(0)
        n += xb.size(0)
    return tot / max(n,1)

@torch.no_grad()
def eval_one(model, loader, device, criterion):
    model.eval()
    tot = 0.0; n = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        tot += loss.item() * xb.size(0); n += xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
    acc = correct / max(n,1)
    return tot / max(n,1), acc

# ----------------------------
# Build train/val item lists
# ----------------------------
def build_items_for_folder(paths, label, val_frac=0.15, max_per_file=None, seed=42):
    rng = np.random.default_rng(seed)
    train_items, val_items = [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            N = f["images"].shape[0]
        rows = np.arange(N)
        if max_per_file and N > max_per_file:
            rows = rng.choice(rows, size=max_per_file, replace=False)
        rng.shuffle(rows)
        n_val = int(len(rows) * val_frac)
        val_rows = rows[:n_val]
        tr_rows = rows[n_val:]
        train_items.append({"path": p, "label": label, "rows": tr_rows})
        val_items.append({"path": p, "label": label, "rows": val_rows})
    return train_items, val_items

# ----------------------------
# Predict on unannotated clusters
# ----------------------------
@torch.no_grad()
def predict_proportions_by_file(model, file_list, device, channel_mean, channel_std, batch_size=256, target_hw=75):
    out = {}
    model.eval()
    for p in file_list:
        with h5py.File(p, "r") as f:
            imgs = f["images"]
            N = imgs.shape[0]
        # stream in batches
        counts = Counter()
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            # stack a batch
            batch = []
            for i in range(start, end):
                with h5py.File(p, "r") as f:
                    arr = f["images"][i].astype(np.float32)
                # resize
                H,W,C = arr.shape
                if H != target_hw or W != target_hw:
                    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
                    t = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
                    arr = t.squeeze(0).permute(1,2,0).numpy()
                # normalize
                for c in range(arr.shape[-1]):
                    arr[..., c] = (arr[..., c] - channel_mean[c]) / (channel_std[c] + 1e-8)
                batch.append(torch.from_numpy(arr).permute(2,0,1))
            xb = torch.stack(batch, dim=0).to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            counts.update(pred)
        out[p] = dict(counts)
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root containing 4 subfolders")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_file", type=int, default=None, help="Optional cap per annotated file")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_dir", type=str, default="models_out")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # label mapping by folder
    LABELS = {"rare_cells_annotated": 0, "junk_annotated": 1, "wbcs_annotated": 2}
    IDX2NAME = {0: "rare", 1: "junk", 2: "wbc"}

    rare_files = find_h5s(args.root, "rare_cells_annotated")
    junk_files = find_h5s(args.root, "junk_annotated")
    wbc_files  = find_h5s(args.root, "wbcs_annotated")
    cluster_files = find_h5s(args.root, "unannotated")

    if not (rare_files or junk_files or wbc_files):
        raise RuntimeError("No annotated files found.")

    # Build items with split
    tr_rare, va_rare = build_items_for_folder(rare_files, LABELS["rare_cells_annotated"], args.val_frac, args.max_per_file, args.seed)
    tr_junk, va_junk = build_items_for_folder(junk_files, LABELS["junk_annotated"], args.val_frac, args.max_per_file, args.seed)
    tr_wbc,  va_wbc  = build_items_for_folder(wbc_files,  LABELS["wbcs_annotated"], args.val_frac, args.max_per_file, args.seed)

    train_items = tr_rare + tr_junk + tr_wbc
    val_items   = va_rare + va_junk + va_wbc

    # Channel stats on training
    print("Estimating channel mean/std from training sample...")
    ch_mean, ch_std = estimate_channel_stats(train_items, max_samples=5000, target_hw=75)
    if ch_mean is None:
        # fallback
        ch_mean = [0.0, 0.0, 0.0, 0.0]
        ch_std  = [1.0, 1.0, 1.0, 1.0]
    print("Channel mean:", [round(x,2) for x in ch_mean])
    print("Channel std :", [round(x,2) for x in ch_std])

    # Datasets
    train_ds = H5ImageDataset(train_items, IDX2NAME, channel_mean=ch_mean, channel_std=ch_std, target_hw=75)
    val_ds   = H5ImageDataset(val_items,   IDX2NAME, channel_mean=ch_mean, channel_std=ch_std, target_hw=75)

    # Weighted sampler to mitigate imbalance
    class_counts = np.zeros(3, dtype=np.int64)
    for it in train_items:
        class_counts[it["label"]] += len(it["rows"])
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    # per-sample weight
    per_class = {i: class_weights[i] for i in range(3)}
    sample_weights = [ per_class[label] for (_, label, _) in train_ds.items ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ------------------ Model A: simple CNN ------------------
    print("\n=== Training Model A: Simple 4-ch CNN ===")
    modelA = SimpleCNN(in_ch=4, n_classes=3).to(device)
    optimizerA = torch.optim.AdamW(modelA.parameters(), lr=args.lr)
    # class-weighted CE
    critA = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    best_val = 1e9; best_pathA = os.path.join(args.save_dir, "cnn_simple.pt")
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        tr_loss = train_one(modelA, train_loader, optimizerA, device, critA, use_amp=args.use_amp)
        va_loss, va_acc = eval_one(modelA, val_loader, device, critA)
        print(f"[A][{ep:02d}/{args.epochs}] train {tr_loss:.4f} | val {va_loss:.4f} | val acc {va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"state_dict": modelA.state_dict(),
                        "mean": ch_mean, "std": ch_std,
                        "label_map": IDX2NAME}, best_pathA)
    print(f"Best A saved to {best_pathA} | time {pretty_time(time.time()-t0)}")

    # ------------------ Model B: ResNet18 (finetuned) ------------------
    print("\n=== Training Model B: ResNet18 4-ch (finetuned) ===")
    modelB = make_resnet18_4ch(n_classes=3, pretrained=True).to(device)
    optimizerB = torch.optim.AdamW(modelB.parameters(), lr=args.lr*0.5)
    critB = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    best_val = 1e9; best_pathB = os.path.join(args.save_dir, "resnet18_4ch.pt")
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        tr_loss = train_one(modelB, train_loader, optimizerB, device, critB, use_amp=args.use_amp)
        va_loss, va_acc = eval_one(modelB, val_loader, device, critB)
        print(f"[B][{ep:02d}/{args.epochs}] train {tr_loss:.4f} | val {va_loss:.4f} | val acc {va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"state_dict": modelB.state_dict(),
                        "mean": ch_mean, "std": ch_std,
                        "label_map": IDX2NAME}, best_pathB)
    print(f"Best B saved to {best_pathB} | time {pretty_time(time.time()-t0)}")

    # ------------------ Predict on unannotated clusters ------------------
    print("\n=== Predicting on unannotated clusters ===")
    # Load bests back (already in memory, but keep it explicit)
    modelA.load_state_dict(torch.load(best_pathA, map_location=device)["state_dict"])
    modelB.load_state_dict(torch.load(best_pathB, map_location=device)["state_dict"])
    modelA.eval(); modelB.eval()

    # proportions per model
    propsA = predict_proportions_by_file(modelA, cluster_files, device, ch_mean, ch_std, batch_size=args.batch_size)
    propsB = predict_proportions_by_file(modelB, cluster_files, device, ch_mean, ch_std, batch_size=args.batch_size)

    def summarize(props, tag):
        print(f"\n--- {tag}: predicted class proportions per cluster ---")
        total_rare = 0; total_all = 0
        for p in sorted(props.keys()):
            fn = os.path.basename(p)
            counts = props[p]
            n = sum(counts.values())
            total_all += n
            rare = counts.get(0,0); junk = counts.get(1,0); wbc = counts.get(2,0)
            total_rare += rare
            print(f"{fn:18s}  n={n:<6d}  rare={rare/n:6.3f}  junk={junk/n:6.3f}  wbc={wbc/n:6.3f}")
        print(f"Overall rare fraction: {total_rare/total_all:.3f}")

        # Quick 3&4 vs others summary
        def getf(idx):
            for p2 in props.keys():
                if p2.endswith(f"MM_cluster_{idx}.hdf5"):
                    c = props[p2]; n2 = sum(c.values()); return c.get(0,0)/max(n2,1)
            return None
        r3 = getf(3); r4 = getf(4)
        r_other = []
        for k in range(0,8):
            if k in (3,4): continue
            r = getf(k)
            if r is not None: r_other.append(r)
        if r3 is not None and r4 is not None and len(r_other)>0:
            print(f"rare_frac(cluster3)={r3:.3f}, rare_frac(cluster4)={r4:.3f}, rare_frac(others_mean)={np.mean(r_other):.3f}")

    summarize(propsA, "Model A (Simple CNN)")
    summarize(propsB, "Model B (ResNet18 4-ch)")

    # Save JSON with raw counts for downstream viz
    out_json = {
        "labels": IDX2NAME,
        "simple_cnn": propsA,
        "resnet18_4ch": propsB
    }
    with open(os.path.join(args.save_dir, "unannotated_pred_counts.json"), "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved raw predictions to {os.path.join(args.save_dir, 'unannotated_pred_counts.json')}")

    # Cleanup file handles
    train_ds.close(); val_ds.close()

if __name__ == "__main__":
    main()
