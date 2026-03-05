from typing import Tuple, Sequence

import h5py
import numpy as np
import pandas as pd
import torch

from junk_qc.utils.io import find_h5s, ensure_dir
from junk_qc.models.factory import build_model


@torch.no_grad()
def infer_file(
    model: torch.nn.Module,
    path: str,
    ch_mean: Sequence[float],
    ch_std: Sequence[float],
    device: str = "cuda",
    batch_size: int = 512,
    target_hw: int = 75,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on all tiles in a single HDF5 file.

    Args:
        model: Trained model in eval mode.
        path: Path to HDF5 file with an `images` dataset.
        ch_mean: Per-channel means for normalization.
        ch_std: Per-channel stds for normalization.
        device: "cuda" or "cpu".
        batch_size: Batch size for inference.
        target_hw: Target height/width for resizing.

    Returns:
        preds: np.ndarray of shape (N,) with hard labels (0/1).
        probs: np.ndarray of shape (N,) with junk probabilities (class 1).
    """
    import torch.nn.functional as F

    model.eval()
    preds: list = []
    probs: list = []
    ch_mean = np.asarray(ch_mean, dtype=np.float32)
    ch_std = np.asarray(ch_std, dtype=np.float32)

    with h5py.File(path, "r") as f:
        imgs = f["images"]
        N = imgs.shape[0]
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            Xb = []
            for i in range(start, end):
                arr = imgs[i].astype(np.float32)
                H, W, C = arr.shape
                if (H != target_hw) or (W != target_hw):
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
                    t = F.interpolate(
                        t,
                        size=(target_hw, target_hw),
                        mode="bilinear",
                        align_corners=False,
                    )
                    arr = t.squeeze(0).permute(1, 2, 0).numpy()
                for c in range(C):
                    arr[..., c] = (arr[..., c] - ch_mean[c]) / (ch_std[c] + 1e-8)
                Xb.append(torch.from_numpy(arr).permute(2, 0, 1))
            xb = torch.stack(Xb, 0).to(device)
            p1 = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            probs.extend(p1.tolist())
            preds.extend((p1 >= 0.5).astype(np.uint8).tolist())
    return np.asarray(preds, np.uint8), np.asarray(probs, np.float32)


def persist_predictions_to_h5(
    path: str,
    pred: np.ndarray,
    prob: np.ndarray,
    model_type: str,
    timestamp: str,
    fold: str,
) -> None:
    """
    Persist predictions into the `/features` table of a HDF5 file.

    Instead of using generic column names like `label` or `score`, this function
    writes:

        label_{model_type}_{timestamp}_{fold}
        score_{model_type}_{timestamp}_{fold}

    where:
        - `model_type` encodes the architecture and augmentation
          (e.g., `simple_cnn_aug`).
        - `timestamp` matches the experiment folder timestamp.
        - `fold` is `1`–`5` for CV models or `all` for the final model.

    Existing columns with the same names are overwritten. Other columns,
    including any ground-truth `label` column, are preserved.

    Args:
        path: Path to the HDF5 file.
        pred: Array of hard predictions (0/1).
        prob: Array of predicted probabilities for class 1.
        model_type: String descriptor of the model type (e.g. `simple_cnn_aug`).
        timestamp: Timestamp string used for the experiment.
        fold: Fold identifier as a string (e.g., "1".."5" or "all").
    """
    try:
        df = pd.read_hdf(path, "features")
    except Exception as e:
        print(f"[persist] {path}: cannot read /features -> {e}")
        return

    n = len(df)
    if len(pred) != n or len(prob) != n:
        print(
            f"[persist] {path}: length mismatch (features={n}, preds={len(pred)}, probs={len(prob)})"
        )
        m = min(n, len(pred), len(prob))
        pred = pred[:m]
        prob = prob[:m]
        df = df.iloc[:m].copy()

    label_col = f"label_{model_type}_{timestamp}_{fold}"
    score_col = f"score_{model_type}_{timestamp}_{fold}"

    extra = pd.DataFrame(
        {
            label_col: np.asarray(pred, dtype=np.uint8),
            score_col: np.asarray(prob, dtype=np.float32),
        },
        index=df.index,
    )

    drop_cols = [c for c in extra.columns if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)

    try:
        with pd.HDFStore(path, mode="a") as store:
            if "/features" in store:
                store.remove("features")
            store.put("features", df, format="table", data_columns=True, index=False)
    except Exception as e:
        print(f"[persist] {path}: write failed -> {e}")


def run_inference_on_subfolder(
    checkpoint_path: str,
    root: str,
    subfolder: str = "unannotated",
    device: str = None,
) -> None:
    """
    Run inference for all HDF5 files in `root/subfolder` using a trained model checkpoint
    and persist predictions into the `/features` table.

    The checkpoint is expected to have been produced by `run_cv_training`, and thus
    contain keys:

        - `arch`
        - `state_dict`
        - `mean`
        - `std`
        - `run_name` (model_type)
        - `timestamp`
        - `fold` (int or "all")

    Args:
        checkpoint_path: Path to a `.pt` file produced by `run_cv_training`.
        root: Root directory containing the subfolder with HDF5 files.
        subfolder: Subfolder name (default "unannotated").
        device: "cuda" or "cpu". If None, chooses automatically.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    arch = ckpt.get("arch", "simple_cnn")
    state_dict = ckpt["state_dict"]
    ch_mean = ckpt["mean"]
    ch_std = ckpt["std"]
    run_name = ckpt.get("run_name", arch)
    timestamp = ckpt.get("timestamp", "unknown")
    fold = ckpt.get("fold", "all")
    if isinstance(fold, int):
        fold = str(fold)

    pretrained = ckpt.get("pretrained", False)
    unfreeze_last = ckpt.get("unfreeze_last", 0)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        arch=arch,
        in_ch=4,
        num_classes=2,
        pretrained=False,  # weights are loaded from checkpoint
        unfreeze_last=unfreeze_last if pretrained else 0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    files = find_h5s(root, subfolder)
    if not files:
        print(f"No HDF5 files found in {root}/{subfolder}")
        return

    for p in files:
        print(f"[infer] {p}")
        preds, probs = infer_file(model, p, ch_mean, ch_std, device=device)
        persist_predictions_to_h5(p, preds, probs, model_type=run_name, timestamp=timestamp, fold=fold)
