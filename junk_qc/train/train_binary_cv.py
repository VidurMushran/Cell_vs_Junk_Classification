from typing import Dict, Any, Tuple, Sequence
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from junk_qc.data.h5_dataset import H5BinaryDataset
from junk_qc.data.splits import make_stratified_kfold_indices, balance_train_items
from junk_qc.data.transforms import get_qualifai_augment
from junk_qc.models.factory import build_model
from junk_qc.models.losses import BinaryFocalLoss
from junk_qc.utils.io import ensure_dir, pretty_time
from contextlib import nullcontext

def _evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, float]:
    """Evaluate model on loader and compute accuracy, ROC-AUC, and F1."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            prob1 = torch.softmax(model(xb), dim=1)[:, 1]
            ys.append(yb.cpu().numpy())
            ps.append(prob1.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    f1 = f1_score(y, pred, zero_division=0)
    return float(acc), float(auc), float(f1)


def _train_single_model(
    train_items,
    val_items,
    ch_mean,
    ch_std,
    arch: str,
    pretrained: bool,
    unfreeze_last: int,
    epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    gamma: float,
    use_amp: bool,
    device: str,
    use_qualifai_aug: bool,
):
    """Train a single model given train/val splits; return model + metrics history."""
    train_transform = get_qualifai_augment(enable=use_qualifai_aug)
    val_transform = None

    train_ds = H5BinaryDataset(train_items, ch_mean, ch_std, transform=train_transform)
    val_ds = H5BinaryDataset(val_items, ch_mean, ch_std, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = build_model(
        arch=arch,
        in_ch=4,
        num_classes=2,
        pretrained=pretrained,
        unfreeze_last=unfreeze_last,
    ).to(device)

    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    try:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler(enabled=use_amp)

        def autocast_ctx():
            # Older torch.cuda.amp.autocast does not take device_type
            if use_amp and device.startswith("cuda"):
                return autocast(enabled=True)
            else:
                return nullcontext()
    except Exception:
        scaler = None

        def autocast_ctx():
            return nullcontext()

    history = {
        "epochs": [],
        "train_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": [],
    }
    best_state = None
    best_metrics = {"acc": -1.0, "auc": float("nan"), "f1": float("nan")}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)

            with autocast_ctx():
                logits = model(xb)
                loss = criterion(logits, yb)

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            bs = xb.size(0)
            tot_loss += loss.item() * bs
            n_samples += bs

        train_loss = tot_loss / max(n_samples, 1)
        acc, auc, f1 = _evaluate(model, val_loader, device=device)

        history["epochs"].append(ep)
        history["train_loss"].append(float(train_loss))
        history["val_acc"].append(float(acc))
        history["val_auc"].append(float(auc))
        history["val_f1"].append(float(f1))

        if acc > best_metrics["acc"]:
            best_metrics = {"acc": float(acc), "auc": float(auc), "f1": float(f1)}
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    dt = pretty_time(time.time() - t0)
    print(
        f"Training done in {dt}. Best val acc={best_metrics['acc']:.3f} auc={best_metrics['auc']:.3f} f1={best_metrics['f1']:.3f}"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_ds.close()
    val_ds.close()

    return model, best_state, history, best_metrics


def run_cv_training(
    items,
    ch_mean,
    ch_std,
    out_dir: str,
    arch: str = "simple_cnn",
    pretrained: bool = False,
    unfreeze_last: int = 0,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    alpha: float = 0.25,
    gamma: float = 2.0,
    use_amp: bool = True,
    device: str = "cuda",
    use_qualifai_aug: bool = True,
    nonjunk_to_junk: float = 1.5,
    n_splits: int = 5,
    seed: int = 42,
    timestamp: str = None,
) -> Dict[str, Any]:
    """
    Run N-fold cross-validation training and a final model on all data.

    For each fold:
      - Balance non-junk:junk ratio in the training items.
      - Train a model and save the best checkpoint + per-fold metrics.
    After all folds:
      - Aggregate metrics (mean/std) across folds and save `cv_metrics_<run_name>.json`.
      - Train a final model on all balanced items and save `final_<run_name>_all_data.pt`.

    Args:
        items: Full list of (path, row_idx, label, meta) tuples.
        ch_mean: Per-channel means for normalization.
        ch_std: Per-channel stds for normalization.
        out_dir: Experiment output directory (created if missing).
        arch: Model architecture (e.g. "simple_cnn", "rot_invariant_cnn", "vgg19_bn").
        pretrained: Whether to use ImageNet-pretrained weights (where supported).
        unfreeze_last: If > 0 and pretrained, number of last layers to unfreeze.
        epochs: Number of training epochs per fold.
        batch_size: Batch size for DataLoader.
        lr: Learning rate for AdamW optimizer.
        alpha: Focal loss alpha parameter.
        gamma: Focal loss gamma parameter.
        use_amp: Whether to use mixed precision (AMP) if CUDA is available.
        device: Device string ("cuda" or "cpu").
        use_qualifai_aug: If True, apply QUALIFAI-style flips/rotations on training set.
        nonjunk_to_junk: Target ratio of non-junk to junk tiles in training.
        n_splits: Number of cross-validation folds (default 5).
        seed: Random seed for splits and balancing.
        timestamp: Timestamp string to tag this experiment (used in checkpoint JSON).

    Returns:
        Dictionary with:
          - run_name
          - timestamp
          - folds: per-fold metrics
          - aggregate: mean/std across folds
          - final_model_path: path to final checkpoint trained on all data
    """
    ensure_dir(out_dir)
    out_dir = Path(out_dir)

    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    run_name = f"{arch}_{'aug' if use_qualifai_aug else 'noaug'}"
    print(f"Run name: {run_name}, timestamp: {timestamp}")

    # 1) Build K-fold splits
    splits = make_stratified_kfold_indices(items, n_splits=n_splits, seed=seed)
    fold_metrics: Dict[str, Dict[str, float]] = {}

    # 2) Cross-validation loops
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")
        raw_train_items = [items[i] for i in tr_idx]
        val_items = [items[i] for i in va_idx]

        balanced_train_items = balance_train_items(
            raw_train_items, nonjunk_to_junk=nonjunk_to_junk, seed=seed + fold_idx
        )
        print(
            f"Fold {fold_idx}: train={len(balanced_train_items)} (balanced), val={len(val_items)}"
        )

        model, best_state, history, best = _train_single_model(
            balanced_train_items,
            val_items,
            ch_mean,
            ch_std,
            arch=arch,
            pretrained=pretrained,
            unfreeze_last=unfreeze_last,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            alpha=alpha,
            gamma=gamma,
            use_amp=use_amp,
            device=device,
            use_qualifai_aug=use_qualifai_aug,
        )

        fold_dir = out_dir / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        # Save checkpoint for this fold
        ckpt_path = fold_dir / f"best_{run_name}_fold{fold_idx}.pt"
        torch.save(
            {
                "arch": arch,
                "pretrained": pretrained,
                "unfreeze_last": unfreeze_last,
                "use_qualifai_aug": use_qualifai_aug,
                "timestamp": timestamp,
                "run_name": run_name,
                "fold": fold_idx,
                "state_dict": best_state,
                "mean": ch_mean,
                "std": ch_std,
                "best_acc": best["acc"],
                "best_auc": best["auc"],
                "best_f1": best["f1"],
            },
            ckpt_path,
        )

        # Save per-fold metrics
        metrics_path = fold_dir / f"metrics_{run_name}_fold{fold_idx}.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "arch": arch,
                    "fold": fold_idx,
                    "timestamp": timestamp,
                    "history": history,
                    "best": best,
                },
                f,
                indent=2,
            )

        fold_metrics[str(fold_idx)] = best

    # 3) Aggregate metrics across folds
    accs = [fold_metrics[str(i)]["acc"] for i in range(1, n_splits + 1)]
    aucs = [fold_metrics[str(i)]["auc"] for i in range(1, n_splits + 1)]
    f1s = [fold_metrics[str(i)]["f1"] for i in range(1, n_splits + 1)]

    aggregate = {
        "mean": {
            "acc": float(np.mean(accs)),
            "auc": float(np.mean(aucs)),
            "f1": float(np.mean(f1s)),
        },
        "std": {
            "acc": float(np.std(accs, ddof=0)),
            "auc": float(np.std(aucs, ddof=0)),
            "f1": float(np.std(f1s, ddof=0)),
        },
    }

    cv_metrics_path = out_dir / f"cv_metrics_{run_name}.json"
    with open(cv_metrics_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "arch": arch,
                "pretrained": pretrained,
                "unfreeze_last": unfreeze_last,
                "use_qualifai_aug": use_qualifai_aug,
                "timestamp": timestamp,
                "n_splits": n_splits,
                "folds": fold_metrics,
                "aggregate": aggregate,
            },
            f,
            indent=2,
        )

    print("\n=== Training final model on all data (balanced) ===")
    all_balanced_items = balance_train_items(
        items, nonjunk_to_junk=nonjunk_to_junk, seed=seed + 999
    )
    final_model, final_state, final_history, final_best = _train_single_model(
        all_balanced_items,
        all_balanced_items,  # use the same set for monitoring (not a true val split)
        ch_mean,
        ch_std,
        arch=arch,
        pretrained=pretrained,
        unfreeze_last=unfreeze_last,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        alpha=alpha,
        gamma=gamma,
        use_amp=use_amp,
        device=device,
        use_qualifai_aug=use_qualifai_aug,
    )

    final_ckpt_path = out_dir / f"final_{run_name}_all_data.pt"
    torch.save(
        {
            "arch": arch,
            "pretrained": pretrained,
            "unfreeze_last": unfreeze_last,
            "use_qualifai_aug": use_qualifai_aug,
            "timestamp": timestamp,
            "run_name": run_name,
            "fold": "all",
            "state_dict": final_state,
            "mean": ch_mean,
            "std": ch_std,
            "best_acc": final_best["acc"],
            "best_auc": final_best["auc"],
            "best_f1": final_best["f1"],
        },
        final_ckpt_path,
    )
    final_metrics_path = out_dir / f"final_metrics_{run_name}_all_data.json"
    with open(final_metrics_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "arch": arch,
                "fold": "all",
                "timestamp": timestamp,
                "history": final_history,
                "best": final_best,
            },
            f,
            indent=2,
        )

    return {
        "run_name": run_name,
        "timestamp": timestamp,
        "folds": fold_metrics,
        "aggregate": aggregate,
        "final_model_path": str(final_ckpt_path),
    }
