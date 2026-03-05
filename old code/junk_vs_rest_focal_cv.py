#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 08:28:40 2025

@author: mushran
"""

# junk_vs_rest_focal_cv.py
# Binary CNN (junk=1 vs rest=0) with focal loss, 10-fold CV.
# Repeats with decreasing train fraction (by moving folds to test) until mean CV accuracy < 0.90.
# Saves:
#   - performance_vs_train_fraction.png
#   - cluster_label_proportions.png
# Writes "label" dataset (uint8) into each unannotated MM_cluster_*.hdf5 (1=junk, 0=not junk).

import os, glob, time, math, json, argparse, random, h5py
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

try:
    # New API for PyTorch >= 2.4
    from torch import amp
    autocast = amp.autocast
    GradScaler = amp.GradScaler
except ImportError:
    # Backward-compatible fallback (older PyTorch)
    from torch.cuda.amp import autocast, GradScaler

# --------------------------
# Utils
# --------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pretty_time(s):
    if s < 60: return f"{s:.1f}s"
    m = int(s//60); return f"{m}m{int(s-60*m)}s"

def find_h5s(root, subfolder):
    d = os.path.join(root, subfolder)
    return sorted(glob.glob(os.path.join(d, "*.hdf5")))

# --------------------------
# Data plumbing
# --------------------------
class H5ImageBinaryDataset(Dataset):
    """
    Streams 4-ch images from multiple HDF5s with a binary label per source file.
    images: (N,H,W,C), uint16/uint8 -> float32, resized to 75x75, normalized by train mean/std.
    """
    def __init__(self, index_list, target_hw=75, ch_mean=None, ch_std=None, dtype=torch.float32):
        """
        index_list: list of tuples (path, row_idx, label) for exact rows to use
        """
        self.items = index_list
        self.target_hw = target_hw
        self.ch_mean = ch_mean
        self.ch_std = ch_std
        self.dtype = dtype
        self._handles = {}  # path -> (h5file, dset)

    def __len__(self):
        return len(self.items)

    def _open(self, path):
        if path in self._handles: return self._handles[path]
        f = h5py.File(path, "r"); d = f["images"]
        self._handles[path] = (f, d); return self._handles[path]

    def _resize_np(self, img):
        H,W,C = img.shape
        if H == self.target_hw and W == self.target_hw: return img
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
        t = F.interpolate(t, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1,2,0).numpy()

    def _normalize(self, arr):
        if self.ch_mean is not None and self.ch_std is not None:
            for c in range(arr.shape[-1]):
                arr[..., c] = (arr[..., c] - self.ch_mean[c]) / (self.ch_std[c] + 1e-8)
            return arr
        # fallback robust scaling
        out = arr.copy()
        for c in range(arr.shape[-1]):
            p = np.percentile(arr[..., c], 99.5); p = max(p,1.0)
            out[..., c] = np.clip(arr[..., c]/p, 0.0, 1.0)
        return out

    def __getitem__(self, idx):
        path, ridx, lab = self.items[idx]
        f, d = self._open(path)
        arr = d[ridx].astype(np.float32)          # (H,W,C)
        arr = self._resize_np(arr)
        arr = self._normalize(arr)
        x = torch.from_numpy(arr).permute(2,0,1).to(self.dtype)  # C,H,W
        y = torch.tensor(lab, dtype=torch.long)
        return x, y

    def close(self):
        for p,(f,_) in self._handles.items():
            try: f.close()
            except: pass
        self._handles.clear()

def build_index_for_files(files, label, max_per_file=None, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for p in files:
        with h5py.File(p, "r") as f:
            N = f["images"].shape[0]
        rows = np.arange(N)
        if max_per_file and N > max_per_file:
            rows = rng.choice(rows, size=max_per_file, replace=False)
        for r in rows:
            out.append((p, int(r), int(label)))
    return out

def estimate_channel_stats(index_list, max_samples=5000, target_hw=75):
    rng = np.random.default_rng(0)
    if len(index_list) == 0: return None, None
    take = min(max_samples, len(index_list))
    sample_idx = rng.choice(np.arange(len(index_list)), size=take, replace=False)
    sums = None; sums2 = None; count = 0; C = None
    for i in sample_idx:
        path, ridx, _ = index_list[i]
        with h5py.File(path, "r") as f:
            arr = f["images"][ridx].astype(np.float32)  # HWC
        # resize
        H,W,Ctmp = arr.shape; C = C or Ctmp
        if H != target_hw or W != target_hw:
            t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
            t = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
            arr = t.squeeze(0).permute(1,2,0).numpy()
        # accumulate channel stats (spatial mean/var per image)
        ch_means = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        ch_vars  = arr.reshape(-1, arr.shape[-1]).var(axis=0)
        if sums is None:
            sums  = np.zeros((arr.shape[-1],), dtype=np.float64)
            sums2 = np.zeros((arr.shape[-1],), dtype=np.float64)
        sums  += ch_means
        sums2 += ch_vars + ch_means**2
        count += 1
    if count == 0: return None, None
    mean = (sums / count).astype(np.float32)
    ex2  = (sums2 / count).astype(np.float32)
    std  = np.sqrt(np.maximum(ex2 - mean**2, 1e-8)).astype(np.float32)
    return mean.tolist(), std.tolist()

# --------------------------
# Model + Focal Loss
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 75->37
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 37->18
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 18->9
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)

class BinaryFocalLoss(nn.Module):
    """
    RetinaNet-style focal loss for logits (before softmax).
    Works for 2-class with targets in {0,1}.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (N,2); target: (N,)
        ce = F.cross_entropy(logits, target, reduction='none')  # -log p_t
        # p_t = prob of the true class
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1)  # (N,)
        alpha_t = torch.where(target==1, self.alpha, 1.0-self.alpha)
        loss = alpha_t * (1-pt).pow(self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

# --------------------------
# Train / Eval
# --------------------------
def run_one_epoch(model, loader, optimizer, device, criterion, use_amp=False):
    model.train(); tot=0.0; n=0
    scaler = GradScaler(enabled=use_amp)
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot/max(n,1)

@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    ys=[]; ps=[]  # prob for class 1
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        prob1 = torch.softmax(logits, dim=1)[:,1]
        ys.append(yb.cpu().numpy()); ps.append(prob1.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p>=0.5).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float('nan')
    f1 = f1_score(y, pred, zero_division=0)
    return acc, auc, f1

def build_fold_indices(index_list, n_splits=10, seed=42):
    # Make arrays for stratified splitting
    y = np.array([lab for (_,_,lab) in index_list], dtype=int)
    idx = np.arange(len(index_list))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in skf.split(idx, y)]
    return idx, folds

# --------------------------
# Label MM_cluster files in-place
# --------------------------
@torch.no_grad()
def label_clusters_inplace(model, files, device, ch_mean, ch_std, batch=512, target_hw=75):
    """
    For each file, compute predicted label (junk=1 vs rest=0) and
    write/overwrite dataset "label" (uint8, shape (N,)).
    Returns {filename: {"n":N, "junk":count1, "notjunk":count0}}
    """
    model.eval()
    out = {}
    for p in files:
        with h5py.File(p, "r+") as f:
            imgs = f["images"]; N = imgs.shape[0]
            labels = np.zeros((N,), dtype=np.uint8)
            for s in range(0, N, batch):
                e = min(N, s+batch)
                Xb = []
                for i in range(s, e):
                    arr = imgs[i].astype(np.float32)  # HWC
                    H,W,C = arr.shape
                    if H != target_hw or W != target_hw:
                        t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
                        t = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
                        arr = t.squeeze(0).permute(1,2,0).numpy()
                    for c in range(C):
                        arr[..., c] = (arr[..., c] - ch_mean[c])/(ch_std[c]+1e-8)
                    Xb.append(torch.from_numpy(arr).permute(2,0,1))
                xb = torch.stack(Xb, dim=0).to(device)
                prob1 = torch.softmax(model(xb), dim=1)[:,1].cpu().numpy()
                labels[s:e] = (prob1 >= 0.5).astype(np.uint8)
            # write/overwrite label dataset
            if "label" in f: del f["label"]
            f.create_dataset("label", data=labels, dtype='uint8')
            c = Counter(labels.tolist())
            out[os.path.basename(p)] = {"n": int(N), "junk": int(c.get(1,0)), "notjunk": int(c.get(0,0))}
    return out

# --------------------------
# Plot helpers
# --------------------------
def plot_perf(curves, out_path):
    # curves: dict with keys 'train_fracs', 'acc', 'auc', 'f1'
    plt.figure(figsize=(8,5))
    x = curves["train_fracs"]
    plt.plot(x, curves["acc"], marker='o', label="Accuracy")
    plt.plot(x, curves["auc"], marker='o', label="AUC")
    plt.plot(x, curves["f1"],  marker='o', label="F1")
    plt.axhline(0.90, linestyle='--')
    plt.xlabel("Train fraction"); plt.ylabel("Score")
    plt.title("CV performance vs training fraction (10-fold baseline, folds moved to test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_cluster_props(cluster_stats, out_path):
    # cluster_stats: {filename: {"n":N, "junk":J, "notjunk":K}}
    names = sorted(cluster_stats.keys())
    junk = [cluster_stats[n]["junk"] / max(cluster_stats[n]["n"],1) for n in names]
    notj = [cluster_stats[n]["notjunk"] / max(cluster_stats[n]["n"],1) for n in names]
    x = np.arange(len(names))
    width = 0.4
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, notj, width=width, label="not junk (0)")
    plt.bar(x + width/2, junk, width=width, label="junk (1)")
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel("Proportion"); plt.title("Predicted label proportions per MM_cluster file")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# --------------------------
# Main driver
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root dir containing subfolders")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_file", type=int, default=None, help="cap rows per annotated file (optional)")
    ap.add_argument("--alpha", type=float, default=0.25, help="Focal Loss alpha (pos class weight)")
    ap.add_argument("--gamma", type=float, default=2.0, help="Focal Loss gamma")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--out_dir", type=str, default="junk_bin_out")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Assemble annotated binary index (junk=1, rare+wbcs=0)
    rare_files = find_h5s(args.root, "rare_cells_annotated")
    wbc_files  = find_h5s(args.root, "wbcs_annotated")
    junk_files = find_h5s(args.root, "junk_annotated")
    if not (rare_files or wbc_files) or not junk_files:
        raise RuntimeError("Need annotated rare/wbc AND junk files.")

    idx_rare = build_index_for_files(rare_files, label=0, max_per_file=args.max_per_file, seed=args.seed)
    idx_wbc  = build_index_for_files(wbc_files,  label=0, max_per_file=args.max_per_file, seed=args.seed)
    idx_junk = build_index_for_files(junk_files, label=1, max_per_file=args.max_per_file, seed=args.seed)
    index_all = idx_rare + idx_wbc + idx_junk
    print(f"Total annotated rows: {len(index_all)}  (not-junk={len(idx_rare)+len(idx_wbc)}, junk={len(idx_junk)})")

    # Channel stats from training universe (computed once)
    ch_mean, ch_std = estimate_channel_stats(index_all, max_samples=6000, target_hw=75)
    if ch_mean is None:
        ch_mean = [0.0,0.0,0.0,0.0]; ch_std = [1.0,1.0,1.0,1.0]
    print("Channel mean:", [round(x,2) for x in ch_mean])
    print("Channel std :", [round(x,2) for x in ch_std])

    # Stratified 10-fold indices (fixed master split)
    master_idx, folds = build_fold_indices(index_all, n_splits=10, seed=args.seed)

    # Progressive train/test: start with 9 folds train / 1 fold test (90/10), then 8/2, ... until <90% acc
    train_fracs=[]; accs=[]; aucs=[]; f1s=[]
    best_model_state = None
    best_ok_train_frac = None

    for n_test_folds in range(1, 10):  # 1..9
        train_folds = list(range(0, 10 - n_test_folds))
        test_folds  = list(range(10 - n_test_folds, 10))
        # Build train/test indices by concatenating the designated folds
        tr_idx = np.concatenate([folds[i][0] for i in train_folds])  # use the train-split part of each chosen fold
        te_idx = np.concatenate([folds[i][1] for i in test_folds])   # use the test-split part of each chosen fold

        # Datasets & loaders
        train_items = [ index_all[i] for i in tr_idx ]
        test_items  = [ index_all[i] for i in te_idx ]
        train_ds = H5ImageBinaryDataset(train_items, ch_mean=ch_mean, ch_std=ch_std)
        test_ds  = H5ImageBinaryDataset(test_items,  ch_mean=ch_mean, ch_std=ch_std)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Model + Focal loss
        model = SimpleCNN(in_ch=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma)

        t0 = time.time()
        for ep in range(1, args.epochs+1):
            tr_loss = run_one_epoch(model, train_loader, optimizer, device, criterion, use_amp=args.use_amp)
            acc, auc, f1 = eval_metrics(model, test_loader, device)
            print(f"[{10-n_test_folds:02d}/10 train folds | ep {ep}/{args.epochs}] "
                  f"loss {tr_loss:.4f} | te acc {acc:.3f} auc {auc:.3f} f1 {f1:.3f}")
        print(f"Fold-config finished in {pretty_time(time.time()-t0)}")

        # Record metrics
        train_frac = (10 - n_test_folds) / 10.0
        train_fracs.append(train_frac); accs.append(acc); aucs.append(auc); f1s.append(f1)

        # Track best-performing model (by accuracy), but still stop if accuracy < 0.90
        if best_model_state is None or acc > best_model_state.get("best_acc", 0):
            best_model_state = {
                "state_dict": model.state_dict(),
                "mean": ch_mean,
                "std": ch_std,
                "train_frac": train_frac,
                "best_acc": acc,
                "best_auc": auc,
                "best_f1": f1,
            }
            best_ok_train_frac = train_frac
        
        # Early stopping condition (once below 0.90 mean acc)
        if acc < 0.90:
            print("Stopping early: accuracy < 0.90.")
            break

        # Clean file handles
        train_ds.close(); test_ds.close()

    # Plot performance vs training fraction
    curves = {"train_fracs": train_fracs, "acc": accs, "auc": aucs, "f1": f1s}
    perf_path = os.path.join(args.out_dir, "performance_vs_train_fraction.png")
    plot_perf(curves, perf_path)
    print(f"Saved performance plot: {perf_path}")

    if best_model_state is None:
        print("No model met the ≥90% accuracy threshold; skipping cluster labeling.")
        return

    # Save best passing model
    model_path = os.path.join(args.out_dir, f"best_binary_model_trainfrac_{int(best_ok_train_frac*100)}.pt")
    torch.save(best_model_state, model_path)
    print(f"Saved best model (acc={best_model_state['best_acc']:.3f}, train_frac={best_model_state['train_frac']:.2f}) to {model_path}")

    # ---------------- Label unannotated clusters in-place + plot proportions ----------------
    cluster_files = find_h5s(args.root, "unannotated")
    if not cluster_files:
        print("No unannotated cluster files found; done.")
        return

    # rebuild model and load best state
    model = SimpleCNN(in_ch=4).to(device)
    model.load_state_dict(best_model_state["state_dict"])

    cluster_stats = label_clusters_inplace(
        model, cluster_files, device,
        ch_mean=best_model_state["mean"], ch_std=best_model_state["std"],
        batch=512, target_hw=75
    )

    # Save JSON stats and plot
    with open(os.path.join(args.out_dir, "cluster_label_counts.json"), "w") as f:
        json.dump(cluster_stats, f, indent=2)
    plot_path = os.path.join(args.out_dir, "cluster_label_proportions.png")
    plot_cluster_props(cluster_stats, plot_path)
    print(f"Wrote label datasets into cluster files and saved plot: {plot_path}")

if __name__ == "__main__":
    main()
