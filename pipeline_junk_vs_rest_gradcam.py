#!/usr/bin/env python3
# pipeline_junk_vs_rest_gradcam.py
# Train junk-vs-rest (unchanged) + run expanded inference & analytics:
# - Inference over all HDF5s in input/unannotated (MM_cluster_* + newly created per-label files)
# - Per-file confusion matrices for non-MM_cluster annotated HDF5s
# - KMeans clustering per annotated label file (v_junk / v_rare / other_junk*) -> save at key 'clusters' with column 'labels'
# - Accuracy metrics within clusters
# - Bar chart: how unlabeled "class 0" (in MM_cluster_*) are classified (junk vs not)
# - Keep Grad-CAM PDFs; add example galleries for correct vs incorrect on annotated junk/rare sets.

import os, glob, time, math, json, argparse, random, h5py, io, re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report

# -----------------------------
# AMP compatibility
# -----------------------------
try:
    from torch import amp
    autocast = amp.autocast
    GradScaler = amp.GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pretty_time(s):
    if s < 60: return f"{s:.1f}s"
    m = int(s//60); return f"{m}m{int(s-60*m)}s"

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def find_h5s(root, subfolder):
    d = os.path.join(root, subfolder)
    return sorted(glob.glob(os.path.join(d, "*.hdf5")))

def is_mm_cluster(fname):
    return os.path.basename(fname).startswith("MM_cluster_")

def sanitize_name(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "_", s)
    return s.strip("._-")

def channels_to_rgb8bit(img_uint16_or_float):
    arr = img_uint16_or_float
    if arr.dtype != np.uint16:
        a = arr.astype(np.float32)
        a = np.clip(a, 0, np.percentile(a, 99.9))
        a = (a / (a.max() + 1e-8) * 65535.0).astype(np.uint16)
    else:
        a = arr
    assert a.shape[-1] >= 3, "Need at least 3 channels (DAPI, TRITC, CY5)"
    rgb16 = np.stack([a[...,1], a[...,2], a[...,0]], axis=-1)  # R=TRITC, G=CY5, B=DAPI
    if a.shape[-1] > 3:
        fitc = a[...,3]
        rgb16 = rgb16 + np.repeat(fitc[...,None], 3, axis=-1)
    rgb16 = np.minimum(rgb16, 65535).astype(np.uint16)
    rgb8  = (rgb16 // 256).astype(np.uint8)
    return rgb8

def make_dapi_gray8(img_uint16_or_float, dapi_idx=0):
    arr = img_uint16_or_float[..., dapi_idx]
    if arr.dtype != np.uint16:
        a = arr.astype(np.float32)
        a = np.clip(a, 0, np.percentile(a, 99.9))
        a = (a / (a.max() + 1e-8) * 65535.0).astype(np.uint16)
    else:
        a = arr
    return (a // 256).astype(np.uint8)

# -----------------------------
# Feature builder for clustering
# -----------------------------
def quick_feats_for_clustering(path, target_hw=75, max_rows=None):
    feats = []
    idxs  = []
    with h5py.File(path, "r") as f:
        X = f["images"]
        N = X.shape[0]
        take = N if max_rows is None else min(max_rows, N)
        for i in range(take):
            arr = X[i].astype(np.float32)   # H,W,C
            H,W,C = arr.shape
            if (H,W)!=(target_hw,target_hw):
                t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
                t = F.interpolate(t, size=(target_hw,target_hw), mode="bilinear", align_corners=False)
                arr = t.squeeze(0).permute(1,2,0).numpy()
            ch_mean = arr.reshape(-1, C).mean(axis=0)
            ch_std  = arr.reshape(-1, C).std(axis=0)
            r = 10
            cx, cy = target_hw//2, target_hw//2
            center = arr[cx-r:cx+r+1, cy-r:cy+r+1, :].reshape(-1, C).mean(axis=0)
            mask = np.ones((target_hw,target_hw), dtype=bool)
            mask[cx-r:cx+r+1, cy-r:cy+r+1] = False
            border = arr[mask].reshape(-1, C).mean(axis=0)
            contrast = center - border
            feats.append(np.concatenate([ch_mean, ch_std, contrast], axis=0))
            idxs.append(i)
    return np.asarray(feats, np.float32), np.asarray(idxs, np.int32)

def choose_k_by_elbow(inertias, ks):
    x1, y1 = ks[0], inertias[0]; x2, y2 = ks[-1], inertias[-1]
    best_k, best_d = ks[0], -1
    for k, y in zip(ks, inertias):
        num = abs((y2 - y1)*k - (x2 - x1)*y + x2*y1 - y2*x1)
        den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2) + 1e-9
        d = num / den
        if d > best_d: best_d, best_k = d, k
    return best_k

# -----------------------------
# Dataset
# -----------------------------
class H5BinaryDataset(Dataset):
    """Streams rows from HDF5 images; returns (tensor, label)."""
    def __init__(self, items, ch_mean, ch_std, target_hw=75, dtype=torch.float32):
        self.items = items
        self.ch_mean = ch_mean; self.ch_std = ch_std
        self.target_hw = target_hw; self.dtype = dtype
        self._handles = {}

    def __len__(self): return len(self.items)

    def _open(self, path):
        if path in self._handles: return self._handles[path]
        f = h5py.File(path, "r")
        d = f["images"]
        self._handles[path] = (f, d)
        return self._handles[path]

    def __getitem__(self, i):
        path, ridx, lab, _ = self.items[i]
        f, d = self._open(path)
        img = d[ridx].astype(np.float32)  # HWC
        H,W,C = img.shape
        if (H,W)!=(self.target_hw,self.target_hw):
            t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
            t = F.interpolate(t, size=(self.target_hw,self.target_hw), mode="bilinear", align_corners=False)
            img = t.squeeze(0).permute(1,2,0).numpy()
        for c in range(img.shape[-1]):
            img[..., c] = (img[..., c] - self.ch_mean[c]) / (self.ch_std[c] + 1e-8)
        x = torch.from_numpy(img).permute(2,0,1).to(self.dtype)  # C,H,W
        y = torch.tensor(int(lab), dtype=torch.long)
        return x, y

    def close(self):
        for p,(f,_) in self._handles.items():
            try: f.close()
            except: pass
        self._handles.clear()

# -----------------------------
# Model + Focal Loss
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 75->37
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 37->18
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 18->9
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc   = nn.Linear(128, 2)

    def forward(self, x):
        x = self.feat(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1)
        alpha_t = torch.where(target==1, self.alpha, 1.0-self.alpha)
        loss = alpha_t * (1-pt).pow(self.gamma) * ce
        if self.reduction=="mean": return loss.mean()
        if self.reduction=="sum":  return loss.sum()
        return loss

# -----------------------------
# Channel stats on annotated
# -----------------------------
def estimate_channel_stats(items, max_samples=6000, target_hw=75):
    rng = np.random.default_rng(0)
    if len(items)==0: return [0,0,0,0],[1,1,1,1]
    take = min(max_samples, len(items))
    picks = rng.choice(np.arange(len(items)), size=take, replace=False)
    sums=None; sums2=None; cnt=0; C=None
    for i in picks:
        path, ridx, _, _ = items[i]
        with h5py.File(path, "r") as f:
            arr = f["images"][ridx].astype(np.float32)
        H,W,Ctmp = arr.shape; C = C or Ctmp
        if (H,W)!=(target_hw,target_hw):
            t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
            t = F.interpolate(t, size=(target_hw,target_hw), mode="bilinear", align_corners=False)
            arr = t.squeeze(0).permute(1,2,0).numpy()
        ch_means = arr.reshape(-1, C).mean(axis=0)
        ch_vars  = arr.reshape(-1, C).var(axis=0)
        if sums is None:
            sums=np.zeros((C,),np.float64); sums2=np.zeros((C,),np.float64)
        sums += ch_means; sums2 += ch_vars + ch_means**2; cnt+=1
    mean = (sums/cnt).astype(np.float32)
    ex2  = (sums2/cnt).astype(np.float32)
    std  = np.sqrt(np.maximum(ex2 - mean**2, 1e-8)).astype(np.float32)
    return mean.tolist(), std.tolist()

# -----------------------------
# Build annotated index + strata (unchanged)
# -----------------------------
def build_annotated_index(root, target_hw=75, junk_k_max=8, junk_max_rows_for_kmeans=None, seed=42):
    rare_files = find_h5s(root, "rare_cells_annotated")
    wbc_files  = find_h5s(root, "wbcs_annotated")
    junk_files = find_h5s(root, "junk_annotated")
    if not junk_files: raise RuntimeError("No junk files found.")
    if not (rare_files or wbc_files): raise RuntimeError("Need rare and/or WBC files.")

    items = []
    for p in rare_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        with h5py.File(p, "r") as f:
            N = f["images"].shape[0]
        for r in range(N): items.append((p, r, 0, f"rare:{stem}"))
    for p in wbc_files:
        with h5py.File(p, "r") as f:
            N = f["images"].shape[0]
        for r in range(N): items.append((p, r, 0, "wbc"))
    junk_rows = []
    for p in junk_files:
        with h5py.File(p, "r") as f:
            N = f["images"].shape[0]
        for r in range(N):
            items.append((p, r, 1, "junk:unknown"))
            junk_rows.append((p, r))

    # KMeans on junk
    feat_list=[]; row_refs=[]
    for p in junk_files:
        feats, idxs = quick_feats_for_clustering(p, target_hw, junk_max_rows_for_kmeans)
        for local_idx in idxs:
            feat_list.append(feats[local_idx]); row_refs.append((p, int(local_idx)))
    X = np.vstack(feat_list).astype(np.float32)
    ks = list(range(2, max(3, junk_k_max+1)))
    inertias=[]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        km.fit(X); inertias.append(km.inertia_)
    k_star = choose_k_by_elbow(inertias, ks)
    km = KMeans(n_clusters=k_star, n_init=20, random_state=seed)
    labels_k = km.fit_predict(X)
    junk_cluster_map = {ref: int(c) for ref,c in zip(row_refs, labels_k)}

    items2=[]
    for (p,r,lab,meta) in items:
        if lab==1:
            c = junk_cluster_map.get((p,r), -1)
            items2.append((p,r,lab, f"junk:c{c}" if c>=0 else "junk:cNA"))
        else:
            items2.append((p,r,lab, meta))
    return items2, k_star

# -----------------------------
# Split & balance (unchanged)
# -----------------------------
def stratified_split_and_balance(items, val_frac=0.40, nonjunk_to_junk=1.5, seed=42):
    rng = np.random.default_rng(seed)
    meta = np.array([it[3] for it in items])
    y    = np.array([it[2] for it in items], int)
    idx  = np.arange(len(items))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(splitter.split(idx, meta))
    train_items = [items[i] for i in tr_idx]
    val_items   = [items[i] for i in va_idx]

    train_y = np.array([it[2] for it in train_items], int)
    nj_idx = [i for i,lab in enumerate(train_y) if lab==0]
    j_idx  = [i for i,lab in enumerate(train_y) if lab==1]
    n_junk = len(j_idx)
    target_nonjunk = int(nonjunk_to_junk * n_junk)

    if len(nj_idx) > target_nonjunk:
        nonjunk_items = [train_items[i] for i in nj_idx]
        nonjunk_meta = np.array([it[3] for it in nonjunk_items])
        counts = Counter(nonjunk_meta.tolist())
        total_nonjunk = len(nonjunk_items)
        selected_idx = []
        for m, cnt in counts.items():
            share = max(1, int(round(target_nonjunk * (cnt / total_nonjunk))))
            sel = rng.choice(np.where(nonjunk_meta==m)[0], size=min(share, cnt), replace=False).tolist()
            selected_idx.extend(sel)
        if len(selected_idx) > target_nonjunk:
            selected_idx = rng.choice(np.array(selected_idx), size=target_nonjunk, replace=False).tolist()
        balanced_nonjunk = [nonjunk_items[i] for i in selected_idx]
        balanced_train = balanced_nonjunk + [train_items[i] for i in j_idx]
    else:
        balanced_train = train_items

    rng.shuffle(balanced_train)
    return balanced_train, val_items

# -----------------------------
# Train / Eval (unchanged)
# -----------------------------
def train_model(train_items, val_items, ch_mean, ch_std, epochs=8, batch_size=256, lr=1e-3, alpha=0.25, gamma=2.0, use_amp=True, device="cuda"):
    train_ds = H5BinaryDataset(train_items, ch_mean, ch_std)
    val_ds   = H5BinaryDataset(val_items,   ch_mean, ch_std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCNN(in_ch=4).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    scaler = GradScaler(enabled=use_amp)
    best = {"acc": -1, "state": None}

    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); tot=0; n=0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            tot += loss.item()*xb.size(0); n += xb.size(0)
        acc, auc, f1 = evaluate(model, val_loader, device)
        print(f"[ep {ep}/{epochs}] train_loss={tot/max(n,1):.4f} | val_acc={acc:.3f} auc={auc:.3f} f1={f1:.3f}")
        if acc > best["acc"]:
            best = {"acc":acc, "auc":auc, "f1":f1, "state":{k:v.cpu() for k,v in model.state_dict().items()}}
    dt = pretty_time(time.time()-t0)
    print(f"Training done in {dt}. Best val acc={best['acc']:.3f} auc={best['auc']:.3f} f1={best['f1']:.3f}")
    model.load_state_dict(best["state"])
    return model, best, (train_ds, val_ds)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys=[]; ps=[]
    for xb,yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        prob1 = torch.softmax(model(xb), dim=1)[:,1]
        ys.append(yb.cpu().numpy()); ps.append(prob1.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = (p>=0.5).astype(int)
    acc = accuracy_score(y, pred)
    try: auc = roc_auc_score(y, p)
    except: auc = float('nan')
    f1 = f1_score(y, pred, zero_division=0)
    return acc, auc, f1

# -----------------------------
# Grad-CAM (unchanged)
# -----------------------------
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.target_module = self.model.feat[-2]  # Conv2d(64,128,...)
        self._acts = None; self._grads = None
        self.h1 = self.target_module.register_forward_hook(self._fw_hook)
        self.h2 = self.target_module.register_full_backward_hook(self._bw_hook)

    def _fw_hook(self, m, inp, out): self._acts = out.detach()
    def _bw_hook(self, m, gin, gout): self._grads = gout[0].detach()

    def _heatmap(self):
        w = self._grads.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * self._acts).sum(dim=1, keepdim=True))
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1,1,1,1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1,1,1,1) + 1e-6
        return (cam - cam_min) / (cam_max - cam_min)

    def generate(self, xb, target_class=1):
        with torch.enable_grad():
            x = xb.detach().clone().requires_grad_(True)
            logits = self.model(x)
            sel = logits[:, target_class].sum()
            self.model.zero_grad(set_to_none=True)
            sel.backward(retain_graph=True)
            return self._heatmap()

    def close(self):
        try: self.h1.remove(); self.h2.remove()
        except: pass

def save_gradcam_pdfs(model, ds, out_dir, ch_mean, ch_std, device="cuda", n_per_class=32, dapi_name_hint="DAPI", pdf_prefix="gradcam_val"):
    ensure_dir(out_dir)

    # Find DAPI channel (best-effort)
    dapi_idx = 0
    try:
        for path, ridx, lab, meta in ds.items[:256]:
            with h5py.File(path,"r") as f:
                if "channels" in f:
                    raw=f["channels"][()]
                    chans=[c.decode() if isinstance(c,(bytes,bytearray)) else str(c) for c in raw]
                    for i,c in enumerate(chans):
                        if c.upper().startswith("DAPI"):
                            dapi_idx=i; raise StopIteration
    except StopIteration:
        pass

    rng = np.random.default_rng(0)
    y = np.array([lab for (_,_,lab,_) in ds.items])
    idx0 = np.where(y==0)[0]; idx1 = np.where(y==1)[0]
    pick0 = rng.choice(idx0, size=min(n_per_class, len(idx0)), replace=False) if len(idx0)>0 else []
    pick1 = rng.choice(idx1, size=min(n_per_class, len(idx1)), replace=False) if len(idx1)>0 else []

    def _make_batch(indices, dapi_only=False):
        xs = []; raws = []
        for i in indices:
            path,ridx,lab,meta = ds.items[i]
            with h5py.File(path,"r") as f:
                raw = f["images"][ridx].astype(np.float32)   # H,W,C
            raws.append(raw.copy())

            img = raw.copy()
            H,W,C = img.shape
            if (H,W)!=(75,75):
                t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
                t = F.interpolate(t, size=(75,75), mode="bilinear", align_corners=False)
                img = t.squeeze(0).permute(1,2,0).numpy()
            for c in range(img.shape[-1]):
                img[...,c] = (img[...,c]-ch_mean[c])/(ch_std[c]+1e-8)
            if dapi_only:
                tmp = np.zeros_like(img); tmp[..., dapi_idx] = img[..., dapi_idx]
                img = tmp
            xs.append(torch.from_numpy(img).permute(2,0,1))
        xb = torch.stack(xs,0).to(device)
        return xb, raws

    class GradCAM:
        def __init__(self, model):
            self.model = model
            self.model.eval()
            self.target_module = self.model.feat[-2]
            self._acts = None; self._grads = None
            self.h1 = self.target_module.register_forward_hook(self._fw_hook)
            self.h2 = self.target_module.register_full_backward_hook(self._bw_hook)
        def _fw_hook(self, m, inp, out): self._acts = out.detach()
        def _bw_hook(self, m, gin, gout): self._grads = gout[0].detach()
        def _heatmap(self):
            w = self._grads.mean(dim=(2,3), keepdim=True)
            cam = F.relu((w * self._acts).sum(dim=1, keepdim=True))
            cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1,1,1,1)
            cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1,1,1,1) + 1e-6
            return (cam - cam_min) / (cam_max - cam_min)
        def generate(self, xb, target_class=1):
            with torch.enable_grad():
                x = xb.detach().clone().requires_grad_(True)
                logits = self.model(x)
                sel = logits[:, target_class].sum()
                self.model.zero_grad(set_to_none=True)
                sel.backward(retain_graph=True)
                return self._heatmap()
        def close(self):
            try: self.h1.remove(); self.h2.remove()
            except: pass

    cammer = GradCAM(model)

    pdf_path = os.path.join(out_dir, f"{pdf_prefix}.pdf")
    with PdfPages(pdf_path) as pdf:
        for group, name in [(pick0, "notjunk"), (pick1, "junk")]:
            if len(group)==0: continue

            xb_comp, raws = _make_batch(group, dapi_only=False)
            cams_comp = cammer.generate(xb_comp, target_class=1).cpu().numpy()[:,0]

            xb_dapi, _ = _make_batch(group, dapi_only=True)
            cams_dapi = cammer.generate(xb_dapi, target_class=1).cpu().numpy()[:,0]

            for k in range(len(group)):
                raw = raws[k]
                rgb = channels_to_rgb8bit(raw)
                dapi8 = make_dapi_gray8(raw, dapi_idx)
                cam_c = cams_comp[k]
                cam_d = cams_dapi[k]

                plt.figure(figsize=(8,8))
                plt.subplot(2,2,1); plt.imshow(rgb);   plt.title(f"{name}: Raw composite"); plt.axis('off')
                plt.subplot(2,2,2); plt.imshow(dapi8, cmap="gray"); plt.title(f"{name}: Raw DAPI"); plt.axis('off')
                plt.subplot(2,2,3); plt.imshow(cam_c, cmap="jet");  plt.title("Grad-CAM (4ch input)"); plt.axis('off')
                plt.subplot(2,2,4); plt.imshow(cam_d, cmap="jet");  plt.title("Grad-CAM (DAPI-only)"); plt.axis('off')
                plt.tight_layout(); pdf.savefig(); plt.close()

    cammer.close()

# -----------------------------
# NEW: Inference utilities
# -----------------------------
@torch.no_grad()
def infer_file(model, path, ch_mean, ch_std, device="cuda", batch=512):
    preds=[]; probs=[]
    with h5py.File(path,"r") as f:
        imgs=f["images"]; N=imgs.shape[0]
        for s in range(0, N, batch):
            e=min(N,s+batch); Xb=[]
            for i in range(s,e):
                arr=imgs[i].astype(np.float32)
                H,W,C=arr.shape
                if (H,W)!=(75,75):
                    t=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
                    t=F.interpolate(t, size=(75,75), mode="bilinear", align_corners=False)
                    arr=t.squeeze(0).permute(1,2,0).numpy()
                for c in range(C):
                    arr[...,c]=(arr[...,c]-ch_mean[c])/(ch_std[c]+1e-8)
                Xb.append(torch.from_numpy(arr).permute(2,0,1))
            xb=torch.stack(Xb,0).to(device)
            p1=torch.softmax(model(xb), dim=1)[:,1].cpu().numpy()
            probs.extend(p1.tolist()); preds.extend((p1>=0.5).astype(np.uint8).tolist())
    return np.array(preds, np.uint8), np.array(probs, np.float32)

def label_from_filename(fname):
    """Return ground-truth 0/1 for annotated per-label files based on name."""
    s = sanitize_name(os.path.basename(fname))
    # v_junk, other_junk* -> 1 ; v_rare, v_wbc -> 0
    if "v_junk" in s: return 1
    if "other_junk" in s or "otherjunk" in s: return 1
    if "v_rare" in s: return 0
    if "v_wbc" in s or "wbc" in s: return 0
    return None  # unknown

def save_confusion_matrix_png(cm, labels, path_png, title):
    plt.figure(figsize=(3.6,3.2))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xticks([0,1],[f"pred {labels[0]}", f"pred {labels[1]}"])
    plt.yticks([0,1],[f"true {labels[0]}", f"true {labels[1]}"])
    for (i,j), z in np.ndenumerate(cm):
        plt.text(j, i, int(z), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

def montage_from_indices(
    h5_path,
    indices,
    out_png,
    max_n=100,
    n_cols=10,
    title=None,
    caption_cols=("slide_id", "frame_id", "cell_id"),
    score_col="model_score",
    pred_col="pred_label",
    font_path="DejaVuSans.ttf",
):
    """
    Create a grid gallery PNG like your prior projects:
      - PIL-based grid (no matplotlib)
      - annotateEZ composite via channels_to_rgb8bit
      - optional title bar
      - optional per-tile captions from /features

    indices: list/np.array of row indices in this H5
    """
    if indices is None or len(indices) == 0:
        return

    rng = np.random.default_rng(0)
    picks = rng.choice(indices, size=min(max_n, len(indices)), replace=False).tolist()

    feat_df = None
    try:
        import pandas as pd
        feat_df = pd.read_hdf(h5_path, "features")
    except Exception:
        feat_df = None

    with h5py.File(h5_path, "r") as f:
        X = f["images"]
        # infer tile size from data
        H, W = X[0].shape[:2]
        tile_size = (W, H)

        tiles = []
        captions = []
        for i in picks:
            try:
                raw = X[int(i)]
            except Exception:
                continue

            rgb = channels_to_rgb8bit(raw)  # exact annotateEZ composite
            tiles.append(Image.fromarray(rgb))

            # caption
            cap = None
            if feat_df is not None and int(i) < len(feat_df):
                row = feat_df.iloc[int(i)]
                parts = []
                for c in caption_cols:
                    if c in row:
                        parts.append(f"{c}:{row[c]}")
                if score_col in row:
                    parts.append(f"score:{float(row[score_col]):.3f}")
                if pred_col in row:
                    parts.append(f"pred:{int(row[pred_col])}")
                cap = " | ".join(parts) if parts else None
            captions.append(cap)

    if len(tiles) == 0:
        return

    n = len(tiles)
    n_rows = (n + n_cols - 1) // n_cols

    # Title bar
    title_h = 0
    if title:
        title_h = 50

    grid_w = n_cols * tile_size[0]
    grid_h = n_rows * tile_size[1]
    gallery = Image.new("RGB", (grid_w, grid_h + title_h), (0, 0, 0))

    # Paste tiles
    for k, tile in enumerate(tiles):
        r = k // n_cols
        c = k % n_cols
        x0 = c * tile_size[0]
        y0 = title_h + r * tile_size[1]
        gallery.paste(tile, (x0, y0))

    draw = ImageDraw.Draw(gallery)

    # Load font
    try:
        font_title = ImageFont.truetype(font_path, 28)
        font_cap   = ImageFont.truetype(font_path, 10)
    except Exception:
        font_title = ImageFont.load_default()
        font_cap   = ImageFont.load_default()

    # Draw title
    if title:
        try:
            bbox = font_title.getbbox(title)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw, th = font_title.getsize(title)
        draw.text(((grid_w - tw)//2, (title_h - th)//2),
                  title, fill=(255, 255, 255), font=font_title)

    # Draw captions lightly in top-left of each tile (optional)
    for k, cap in enumerate(captions):
        if not cap:
            continue
        r = k // n_cols
        c = k % n_cols
        x0 = c * tile_size[0] + 2
        y0 = title_h + r * tile_size[1] + 2
        draw.text((x0, y0), cap, fill=(255, 255, 255), font=font_cap)

    ensure_dir(os.path.dirname(out_png))
    gallery.save(out_png)

def file_gt_from_name(path):
    """Map per-label filenames to ground truth 0/1 where possible."""
    s = sanitize_name(os.path.basename(path))
    if "v_junk" in s: return 1
    if "other_junk" in s or "otherjunk" in s: return 1
    if "v_rare" in s: return 0
    if "v_wbc" in s or "wbc" in s: return 0
    return None

def persist_predictions_to_h5(path, pred, prob, overwrite_label=False):
    """
    Writes into /features:
      - pred_label (uint8)
      - confidence_score (float32)
      - model_score (float32)  [only for v_junk/v_rare files]
      - gt_label (int, when inferrable from filename)
      - label_text (object) like "1" (junk) or "Class: 0" (non-junk) for UI
    NEVER converts df['label'] away from integer dtype (to keep annotateEZ working).
    If an older file has string labels, it will coerce them back to 0/1 safely.
    """
    import pandas as pd

    try:
        df = pd.read_hdf(path, "features")
    except Exception as e:
        print(f"[persist] {os.path.basename(path)}: cannot read features -> {e}")
        return

    n = len(df)
    if len(pred) != n or len(prob) != n:
        print(f"[persist] {os.path.basename(path)}: length mismatch (features={n}, preds={len(pred)})")
        m = min(n, len(pred), len(prob))
        pred = pred[:m]; prob = prob[:m]
        df = df.iloc[:m].copy()

    # --- Repair legacy string label columns if needed ---
    if "label" in df.columns and not pd.api.types.is_integer_dtype(df["label"]):
        # map common legacy forms back to ints
        _s = df["label"].astype(str).str.strip().str.lower()
        repaired = np.full(len(df), 0, dtype=np.uint8)
        repaired[_s.eq("1")] = 1
        repaired[_s.eq("junk")] = 1
        repaired[_s.eq("class: 1")] = 1
        repaired[_s.eq("class: 0")] = 0
        repaired[_s.eq("0")] = 0
        df["label"] = repaired.astype(np.uint8)

    # Build in one shot to avoid fragmentation warnings
    extra = pd.DataFrame({
        "pred_label":       np.asarray(pred, dtype=np.uint8),
        "confidence_score": np.asarray(prob, dtype=np.float32),
        "label_text":       np.where(np.asarray(pred)==1, "1", "Class: 0").astype(object),
    }, index=df.index)

    # Ground truth (by filename)
    gt = file_gt_from_name(path)
    if gt is not None:
        extra["gt_label"] = int(gt)

    # model_score only for v_junk / v_rare
    base = sanitize_name(os.path.basename(path))
    if ("v_junk" in base) or ("v_rare" in base):
        extra["model_score"] = np.asarray(prob, dtype=np.float32)

    # Overwrite numeric 'label' ONLY if you explicitly asked, and ONLY as ints
    if overwrite_label and "label" in df.columns and pd.api.types.is_integer_dtype(df["label"]):
        df["label"] = np.asarray(pred, dtype=df["label"].dtype)

    # Drop dup cols then concat
    drop_cols = [c for c in extra.columns if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)

    # Write back atomically
    try:
        with pd.HDFStore(path, mode="a") as store:
            if "/features" in store:
                store.remove("features")
            store.put("features", df, format="table", data_columns=True, index=False)
    except Exception as e:
        print(f"[persist] {os.path.basename(path)}: write failed -> {e}")

def plot_cluster_metrics(json_path, out_png):
    with open(json_path, "r") as f:
        data = json.load(f)
    met = data.get("per_cluster", {})
    if not met:
        return
    ks   = sorted(int(k) for k in met.keys())
    accs = [met[str(k)]["acc"] for k in ks]
    f1s  = [met[str(k)]["f1"]  for k in ks]
    n    = [met[str(k)]["n"]   for k in ks]
    x = np.arange(len(ks))
    plt.figure(figsize=(7,4))
    plt.plot(x, accs, marker="o", label="Accuracy")
    plt.plot(x, f1s,  marker="s", label="F1")
    plt.xticks(x, [f"c{k}\n(n={n[i]})" for i,k in enumerate(ks)])
    plt.ylim(0,1); plt.ylabel("Score"); plt.title("Per-cluster metrics")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def cluster_galleries_per_label_file(h5_path, out_dir, target_hw=75):
    try:
        ctab = pd.read_hdf(h5_path, "clusters")
    except Exception:
        return

    ensure_dir(out_dir)
    for cl in sorted(ctab["labels"].unique()):
        idx = np.where(ctab["labels"].values == cl)[0]
        out_png = os.path.join(
            out_dir,
            f"{os.path.splitext(os.path.basename(h5_path))[0]}_cluster{cl}.png"
        )
        montage_from_indices(
            h5_path, idx.tolist(), out_png,
            max_n=100, n_cols=10,
            title=f"{os.path.basename(h5_path)} | cluster {cl}"
        )

def bar_mm_clusters_only(model, root, ch_mean, ch_std, out_dir, device="cuda"):
    files = [p for p in find_h5s(root, "unannotated") if is_mm_cluster(p)]
    ensure_dir(out_dir)
    stats = {}
    for p in files:
        preds, _ = infer_file(model, p, ch_mean, ch_std, device=device, batch=512)
        c = Counter(preds.tolist())
        stats[os.path.basename(p)] = {"n": int(len(preds)),
                                      "junk": int(c.get(1,0)),
                                      "cells": int(c.get(0,0))}
    if not stats:
        print("[mmbar] No MM_cluster_*.hdf5 files found.")
        return
    names = sorted(stats.keys())
    junk_counts  = [stats[n]["junk"]  for n in names]
    cell_counts  = [stats[n]["cells"] for n in names]
    x = np.arange(len(names)); width=0.4
    plt.figure(figsize=(12,5))
    plt.bar(x-width/2, cell_counts, width=width, label="pred cells (0)")
    plt.bar(x+width/2, junk_counts, width=width, label="pred junk (1)")
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel("Count"); plt.title("Predictions per MM_cluster file (annotated sets excluded)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mm_clusters_pred_counts.png"), dpi=150)
    plt.close()
    with open(os.path.join(out_dir, "mm_clusters_pred_counts.json"), "w") as f:
        json.dump(stats, f, indent=2)

from sklearn.metrics import roc_curve, auc

def roc_from_vjunk_vrare(root, ch_mean, ch_std, model, device, out_dir):
    ensure_dir(out_dir)
    pos_files = [p for p in find_h5s(root, "unannotated") if "v_junk" in sanitize_name(os.path.basename(p))]
    neg_files = [p for p in find_h5s(root, "unannotated") if "v_rare" in sanitize_name(os.path.basename(p))]

    y = []; scores = []
    for p in pos_files:
        _, prob = infer_file(model, p, ch_mean, ch_std, device=device)
        y.append(np.ones_like(prob)); scores.append(prob)
        # ensure model_score lives in features too
        persist_predictions_to_h5(p, (prob>=0.5).astype(np.uint8), prob, overwrite_label=False)
    for p in neg_files:
        _, prob = infer_file(model, p, ch_mean, ch_std, device=device)
        y.append(np.zeros_like(prob)); scores.append(prob)
        persist_predictions_to_h5(p, (prob>=0.5).astype(np.uint8), prob, overwrite_label=False)

    if not y:
        print("[roc] No v_junk/v_rare files found.")
        return

    y = np.concatenate(y).astype(np.uint8)
    scores = np.concatenate(scores).astype(np.float32)
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], lw=1, ls="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC: v_junk (pos) vs v_rare (neg)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_vjunk_vs_vrare.png"), dpi=180)
    plt.close()

    # Save raw arrays
    np.savez(os.path.join(out_dir, "roc_vjunk_vs_vrare.npz"), fpr=fpr, tpr=tpr, auc=roc_auc)
    
# -----------------------------
# NEW: Clustering per labeled file -> save to key 'clusters' (labels column)
# -----------------------------
def cluster_labeled_file(path, out_dir, preds=None, y_true=None, target_hw=75, seed=42):
    feats, idxs = quick_feats_for_clustering(path, target_hw=target_hw, max_rows=None)
    ks = [2,3,4,5,6]
    inertias=[]
    for k in ks:
        km=KMeans(n_clusters=k, n_init=10, random_state=seed)
        km.fit(feats); inertias.append(km.inertia_)
    k_star = choose_k_by_elbow(inertias, ks)
    km=KMeans(n_clusters=k_star, n_init=20, random_state=seed)
    c_labels = km.fit_predict(feats)

    # Save clusters table under key 'clusters' with DataFrame column 'labels'
    import pandas as pd
    df = pd.DataFrame({"labels": c_labels.astype(np.int32)})
    with pd.HDFStore(path, mode="a") as store:
        store.put("clusters", df, format="table", data_columns=True, index=False)

    # If we have preds + y_true for this file, report accuracy within clusters
    metrics = {}
    if preds is not None and y_true is not None and len(preds)==len(c_labels):
        for cl in sorted(np.unique(c_labels)):
            m = c_labels==cl
            if m.sum()==0: continue
            acc = accuracy_score(y_true[m], preds[m])
            f1  = f1_score(y_true[m], preds[m], zero_division=0)
            metrics[int(cl)] = {"n": int(m.sum()), "acc": float(acc), "f1": float(f1)}

        # Save JSON summary
        bname = os.path.splitext(os.path.basename(path))[0]
        ensure_dir(out_dir)
        with open(os.path.join(out_dir, f"{bname}_cluster_metrics.json"), "w") as f:
            json.dump({"k": int(k_star), "per_cluster": metrics}, f, indent=2)

    return k_star, metrics

# -----------------------------
# NEW: Post-inference analytics driver
# -----------------------------
def post_inference_analytics(model, root, ch_mean, ch_std, out_dir, device="cuda", overwrite_label=False):
    ensure_dir(out_dir)
    una_dir = os.path.join(root, "unannotated")
    files = sorted(glob.glob(os.path.join(una_dir, "*.hdf5")))
    if not files:
        print("No HDF5 files found in 'unannotated/'.")
        return

    file_preds = {}
    file_probs = {}

    # 1) Inference on all files, then persist predictions into features
    for p in files:
        print(f"[infer] {os.path.basename(p)}")
        preds, probs = infer_file(model, p, ch_mean, ch_std, device=device, batch=512)
        file_preds[p] = preds
        file_probs[p] = probs
        persist_predictions_to_h5(p, preds, probs, overwrite_label=overwrite_label)

    # 2) Confusion matrices for non-MM_cluster files (per-file)
    #    Ground-truth by filename mapping (v_junk/other_junk => 1, v_rare/v_wbc => 0)
    cm_dir = os.path.join(out_dir, "confusion_matrices"); ensure_dir(cm_dir)
    txt_dir = os.path.join(out_dir, "reports"); ensure_dir(txt_dir)
    for p in files:
        if is_mm_cluster(p):  # skip
            continue
        gt = label_from_filename(p)
        if gt is None:  # unknown file type
            continue
        y_true = np.full_like(file_preds[p], fill_value=gt)
        y_pred = file_preds[p]
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        bname = os.path.splitext(os.path.basename(p))[0]
        save_confusion_matrix_png(cm, labels=[0,1],
                                  path_png=os.path.join(cm_dir, f"{bname}_cm.png"),
                                  title=f"{bname} (per-file)")
        # Also a text report
        report = classification_report(y_true, y_pred, labels=[0,1], target_names=["notjunk(0)","junk(1)"], zero_division=0)
        with open(os.path.join(txt_dir, f"{bname}_report.txt"), "w") as f:
            f.write(report)

    # 3) Cluster each labeled file (v_junk / v_rare / other_junk*) and report accuracy within clusters
    clus_dir = os.path.join(out_dir, "cluster_metrics"); ensure_dir(clus_dir)
    for p in files:
        if is_mm_cluster(p):  # only per-label files
            continue
        gt = label_from_filename(p)
        if gt is None:  # skip unknown
            continue
        y_true = np.full_like(file_preds[p], fill_value=gt)
        y_pred = file_preds[p]
        try:
            cluster_labeled_file(p, out_dir=clus_dir, preds=y_pred, y_true=y_true, target_hw=75, seed=42)
        except Exception as e:
            print(f"[cluster] Failed on {os.path.basename(p)}: {e}")

    # 4) Bar graph: among MM_cluster_* rows that are 'class 0' (unlabeled), what did model predict?
    bar_dir = os.path.join(out_dir, "unlabeled_bar"); ensure_dir(bar_dir)
    unlabeled_stats = {}
    for p in files:
        if not is_mm_cluster(p): continue
        # Need features['labels'] equal to 'class 0' (case-insensitive)
        try:
            import pandas as pd
            df = pd.read_hdf(p, "features")
            if "labels" not in df.columns:
                continue
            s = df["labels"].astype(str).str.strip().str.lower()
            mask = (s == "class 0")
            if not mask.any():
                continue
            preds = file_preds[p][mask.to_numpy()]
            n = int(mask.sum())
            unlabeled_stats[os.path.basename(p)] = {
                "n_class0": n,
                "junk": int((preds==1).sum()),
                "notjunk": int((preds==0).sum())
            }
        except Exception:
            # If features absent or unreadable, skip this file
            continue
        
        bname = os.path.splitext(os.path.basename(p))[0]
        json_path = os.path.join(clus_dir, f"{bname}_cluster_metrics.json")
        
        if os.path.exists(json_path):
            plot_cluster_metrics(json_path, os.path.join(clus_dir, f"{bname}_cluster_metrics.png"))
        
        cluster_galleries_per_label_file(p, out_dir=os.path.join(clus_dir, "galleries"))
    
    if unlabeled_stats:
        names = sorted(unlabeled_stats.keys())
        junk = [unlabeled_stats[n]["junk"]/max(unlabeled_stats[n]["n_class0"],1) for n in names]
        notj = [unlabeled_stats[n]["notjunk"]/max(unlabeled_stats[n]["n_class0"],1) for n in names]
        x = np.arange(len(names)); width=0.4
        plt.figure(figsize=(10,5))
        plt.bar(x-width/2, notj, width=width, label="pred not junk (0)")
        plt.bar(x+width/2, junk,  width=width, label="pred junk (1)")
        plt.xticks(x, names, rotation=45, ha='right')
        plt.ylabel("Proportion")
        plt.title("Unlabeled 'class 0' predictions by file")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(bar_dir, "class0_pred_bar.png"), dpi=150)
        plt.close()
        with open(os.path.join(bar_dir, "class0_pred_counts.json"), "w") as f:
            json.dump(unlabeled_stats, f, indent=2)

    # 5) Example galleries: annotated junk & rare files (correct vs incorrect)
    gal_dir = os.path.join(out_dir, "galleries"); ensure_dir(gal_dir)
    for p in files:
        if is_mm_cluster(p): continue
        gt = label_from_filename(p)
        if gt is None: continue
        y_pred = file_preds[p]
        bname = os.path.splitext(os.path.basename(p))[0]
        # tp/fp/tn/fn indices
        if gt == 1:
            tp = np.where(y_pred == 1)[0]
            fn = np.where(y_pred == 0)[0]
            montage_from_indices(
                p, tp.tolist(),
                os.path.join(gal_dir, f"{bname}_CORRECT_junk_TP.png"),
                max_n=100, n_cols=10,
                title=f"{bname} | TP (junk correct)"
            )
            montage_from_indices(
                p, fn.tolist(),
                os.path.join(gal_dir, f"{bname}_INCORRECT_junk_FN.png"),
                max_n=100, n_cols=10,
                title=f"{bname} | FN (junk missed)"
            )
        else:
            tn = np.where(y_pred == 0)[0]
            fp = np.where(y_pred == 1)[0]
            montage_from_indices(
                p, tn.tolist(),
                os.path.join(gal_dir, f"{bname}_CORRECT_rare_TN.png"),
                max_n=100, n_cols=10,
                title=f"{bname} | TN (rare correct)"
            )
            montage_from_indices(
                p, fp.tolist(),
                os.path.join(gal_dir, f"{bname}_INCORRECT_rare_FP.png"),
                max_n=100, n_cols=10,
                title=f"{bname} | FP (rare flagged junk)"
            )

# -----------------------------
# Predict on unannotated clusters (original plot, kept)
# -----------------------------
@torch.no_grad()
def predict_unannotated_and_plot(model, root, ch_mean, ch_std, out_dir, device="cuda"):
    files = find_h5s(root, "unannotated")
    if not files:
        print("No unannotated files found.")
        return
    stats = {}
    for p in files:
        preds, _ = infer_file(model, p, ch_mean, ch_std, device=device, batch=512)
        c = Counter(preds.tolist())
        stats[os.path.basename(p)] = {"n":int(len(preds)), "junk":int(c.get(1,0)), "notjunk":int(c.get(0,0))}
    names = sorted(stats.keys())
    junk = [stats[n]["junk"]/max(stats[n]["n"],1) for n in names]
    notj = [stats[n]["notjunk"]/max(stats[n]["n"],1) for n in names]
    x = np.arange(len(names)); width=0.4
    plt.figure(figsize=(10,5))
    plt.bar(x-width/2, notj, width=width, label="not junk (0)")
    plt.bar(x+width/2, junk,  width=width, label="junk (1)")
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel("Proportion"); plt.title("Predicted label proportions per file")
    plt.legend(); plt.tight_layout()
    ensure_dir(out_dir)
    plot_path = os.path.join(out_dir, "unannotated_label_proportions.png")
    plt.savefig(plot_path, dpi=150); plt.close()
    with open(os.path.join(out_dir,"unannotated_label_counts.json"),"w") as f:
        json.dump(stats, f, indent=2)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root containing subfolders: junk_annotated, rare_cells_annotated, wbcs_annotated, unannotated")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.25, help="Focal Loss alpha (pos class weight)")
    ap.add_argument("--gamma", type=float, default=2.0,  help="Focal Loss gamma")
    ap.add_argument("--nonjunk_to_junk", type=float, default=1.5, help="Train downsampling ratio (nonjunk : junk)")
    ap.add_argument("--junk_k_max", type=int, default=8, help="Max K to try for KMeans elbow (junk stratification during training)")
    ap.add_argument("--junk_max_rows_for_kmeans", type=int, default=None, help="Cap rows per junk file when stratifying for training (speed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--out_dir", type=str, default="pipeline_out")
    ap.add_argument("--overwrite_label", action="store_true", help="If set, overwrite numeric 'label' column in features with model predictions (0/1).")
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.out_dir == "pipeline_out" or not args.out_dir:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"experiment_outputs_{timestr}"

    ensure_dir(args.out_dir)

    # ===== TRAINING (unchanged) =====
    print("Building annotated index & junk clusters for training...")
    items, k_star = build_annotated_index(
        args.root,
        target_hw=75,
        junk_k_max=args.junk_k_max,
        junk_max_rows_for_kmeans=args.junk_max_rows_for_kmeans,
        seed=args.seed
    )
    print(f"Chosen K for junk clusters (training stratification): {k_star}")

    ch_mean, ch_std = estimate_channel_stats(items, max_samples=6000, target_hw=75)
    print("Channel mean:", [round(x,2) for x in ch_mean])
    print("Channel std :", [round(x,2) for x in ch_std])

    train_items, val_items = stratified_split_and_balance(
        items, val_frac=0.40, nonjunk_to_junk=args.nonjunk_to_junk, seed=args.seed
    )
    print(f"Train size: {len(train_items)}, Val size: {len(val_items)}")

    model, best, (train_ds, val_ds) = train_model(
        train_items, val_items, ch_mean, ch_std,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        alpha=args.alpha, gamma=args.gamma, use_amp=args.use_amp, device=device
    )
    torch.save({
        "state_dict": {k:v.cpu() for k,v in model.state_dict().items()},
        "mean": ch_mean, "std": ch_std,
        "best_acc": best["acc"], "best_auc": best["auc"], "best_f1": best["f1"]
    }, os.path.join(args.out_dir, "best_binary_cnn.pt"))

    # Grad-CAM on validation (kept)
    print("Generating Grad-CAM PDFs on validation set...")
    save_gradcam_pdfs(model, val_ds, args.out_dir, ch_mean, ch_std, device=device,
                      n_per_class=32, pdf_prefix="gradcam_val")
    train_ds.close(); val_ds.close()

    # ===== NEW POST-INFERENCE ANALYTICS =====
    print("Running full-directory inference and analytics...")
    post_inference_analytics(
        model, args.root, ch_mean, ch_std,
        out_dir=os.path.join(args.out_dir, "analytics"),
        device=device,
        overwrite_label=args.overwrite_label
    )
    
    # Extra analytics
    roc_from_vjunk_vrare(args.root, ch_mean, ch_std, model, device, out_dir=os.path.join(args.out_dir, "analytics"))
    bar_mm_clusters_only(model, args.root, ch_mean, ch_std, out_dir=os.path.join(args.out_dir, "analytics", "mm_clusters_only"), device=device)
    predict_unannotated_and_plot(model, args.root, ch_mean, ch_std, out_dir=os.path.join(args.out_dir, "analytics"), device=device)

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
