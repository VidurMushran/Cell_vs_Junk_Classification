#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-slide top-k outlier scorer using ECOD and COPOD (pyod).

What this version handles:
- Reads parquet with fastparquet (since pyarrow fails on these source files)
- Decodes feature columns z0..z127 when they are stored as raw 2-byte float16 bytes
- Casts features to float32 for modeling
- Scores all rows at once with ECOD and COPOD
- Saves a histogram per method with a vertical line at the score threshold for top_n_threshold
- Writes a parquet containing the top_k rows per method:
    <output_dir>/<parquet_stem>_{method}_{top_k}.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD

# Fixed feature columns z0..z127
FEAT_COLS = [f"z{i}" for i in range(128)]
MORPH_COLS = ["area", "eccentricity", "DAPI_mean", "TRITC_mean", "CY5_mean", "FITC_mean"]
BASE_METADATA_COLS = ["frame_id", "cell_id", "x", "y"]


def load_config(path: Path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def get_available_columns_fast(parquet_path: Path):
    """
    Read a small sample with fastparquet so we can discover available columns
    without relying on pyarrow.
    """
    df0 = pd.read_parquet(parquet_path, engine="fastparquet")
    return list(df0.columns)


def read_parquet(parquet_path: Path, feat_cols):
    """
    Read parquet with fastparquet because pyarrow fails on the current source files.
    Restrict to known columns if they exist.
    """
    available_cols = get_available_columns_fast(parquet_path)

    required_missing = [c for c in feat_cols if c not in available_cols]
    if required_missing:
        raise RuntimeError(f"Missing feature columns in parquet: {required_missing}")

    desired_cols = feat_cols + MORPH_COLS + BASE_METADATA_COLS
    read_cols = [c for c in desired_cols if c in available_cols]

    df = pd.read_parquet(parquet_path, engine="fastparquet", columns=read_cols)
    return df


def decode_float16_byte_series(s: pd.Series) -> np.ndarray:
    """
    Decode a Series whose entries are raw 2-byte little-endian float16 values,
    returned by fastparquet as Python bytes objects.
    Returns float32 numpy array.
    """
    vals = s.to_numpy(dtype=object, copy=False)
    mask = pd.isna(vals)

    out = np.full(len(vals), np.nan, dtype=np.float32)
    nonnull = vals[~mask]

    if len(nonnull) == 0:
        return out

    first = nonnull[0]
    if isinstance(first, memoryview):
        first = first.tobytes()

    # Fast path: all entries are raw 2-byte values
    if isinstance(first, (bytes, bytearray)) and len(first) == 2:
        try:
            raw = b"".join(
                x.tobytes() if isinstance(x, memoryview) else bytes(x)
                for x in nonnull
            )
            out[~mask] = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            return out
        except Exception:
            pass

    # Fallback path for mixed/object content
    decoded = []
    for x in nonnull:
        if isinstance(x, memoryview):
            x = x.tobytes()

        if isinstance(x, (bytes, bytearray)):
            b = bytes(x)
            if len(b) == 2:
                try:
                    decoded.append(float(np.frombuffer(b, dtype="<f2")[0]))
                    continue
                except Exception:
                    decoded.append(np.nan)
                    continue
            elif len(b) == 4:
                try:
                    decoded.append(float(np.frombuffer(b, dtype="<f4")[0]))
                    continue
                except Exception:
                    decoded.append(np.nan)
                    continue
            else:
                try:
                    decoded.append(float(b.decode("utf-8")))
                    continue
                except Exception:
                    decoded.append(np.nan)
                    continue
        else:
            decoded.append(x)

    out[~mask] = pd.to_numeric(np.array(decoded, dtype=object), errors="coerce").astype(np.float32)
    return out


def safe_cast_features(df: pd.DataFrame, feat_cols):
    """
    Converts feature columns into float32.
    Handles object/bytes columns from fastparquet by decoding raw float16 bytes.
    Drops rows with missing feature values afterward.
    """
    df_clean = df.copy()

    object_cols = [c for c in feat_cols if df_clean[c].dtype == object]
    if object_cols:
        logging.info(f"Decoding {len(object_cols)} object-typed feature columns from raw bytes")

    for col in feat_cols:
        s = df_clean[col]
        if s.dtype == object:
            df_clean[col] = decode_float16_byte_series(s)
        else:
            df_clean[col] = pd.to_numeric(s, errors="coerce").astype(np.float32)

    before = len(df_clean)
    df_clean = df_clean.dropna(subset=feat_cols).copy()
    after = len(df_clean)

    if after == 0:
        raise RuntimeError("All rows dropped after cleaning feature columns.")

    dropped = before - after
    if dropped > 0:
        logging.info(f"Dropped {dropped} rows after feature cleaning/dropna")

    return df_clean


def fit_and_score_ecod(X: np.ndarray):
    model = ECOD()
    model.fit(X)
    return model, np.asarray(model.decision_scores_)


def fit_and_score_copod(X: np.ndarray):
    model = COPOD()
    model.fit(X)
    return model, np.asarray(model.decision_scores_)


def save_histogram(
    scores: np.ndarray,
    threshold_value: float,
    top_n_threshold: int,
    out_path: Path,
    title: str,
):
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=100)
    ax.axvline(
        threshold_value,
        linestyle="--",
        linewidth=2,
        label=f"threshold (top {top_n_threshold})",
    )
    ax.set_title(title)
    ax.set_xlabel("Outlier score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def topk_indices(scores: np.ndarray, k: int):
    n = len(scores)
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def maybe_log_feature_diagnostics(df: pd.DataFrame, feat_cols):
    object_cols = [c for c in feat_cols if df[c].dtype == object]
    logging.info(f"Feature columns read: {len(feat_cols)} total")
    logging.info(f"Object-typed feature columns before cleaning: {len(object_cols)}")
    if object_cols:
        sample_col = object_cols[0]
        sample_vals = df[sample_col].dropna().head(5).tolist()
        logging.info(f"Sample raw values from {sample_col}: {sample_vals}")


def main():
    p = argparse.ArgumentParser(description="Single-slide top-k outlier scoring (ECOD, COPOD).")
    p.add_argument("--config", "-c", required=True, help="YAML config file path.")
    p.add_argument("--parquet", "-p", required=False, help="Path to slide parquet (overrides config).")
    p.add_argument("--override_topk", type=int, default=None, help="Override top_k from config.")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(cfg_path)

    parquet_cli = Path(args.parquet) if args.parquet else None
    parquet_cfg = Path(cfg["parquet_path"]) if cfg.get("parquet_path") else None

    if parquet_cli:
        parquet_path = parquet_cli
    elif parquet_cfg:
        parquet_path = parquet_cfg
    else:
        print("ERROR: parquet path must be provided via --parquet or 'parquet_path' in config.", file=sys.stderr)
        sys.exit(2)

    if not parquet_path.exists():
        print(f"Parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(2)

    methods = [m.lower() for m in cfg.get("methods", ["ecod", "copod"])]
    top_k = args.override_topk if args.override_topk is not None else int(cfg.get("top_k", 10000))
    top_n_threshold = int(cfg.get("top_n_threshold", 2500))
    out_dir = Path(cfg.get("output_dir", "."))
    hist_subdir = cfg.get("hist_subdir", "histograms")

    ensure_dir(out_dir)
    hist_dir = out_dir / hist_subdir
    ensure_dir(hist_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info(f"Reading parquet with fastparquet: {parquet_path}")
    df = read_parquet(parquet_path, FEAT_COLS)
    maybe_log_feature_diagnostics(df, FEAT_COLS)

    metadata_cols = [c for c in df.columns if c not in FEAT_COLS]
    if "slide_id" not in df.columns:
        df["slide_id"] = parquet_path.stem
        metadata_cols.append("slide_id")

    df = safe_cast_features(df, FEAT_COLS)
    X = df[FEAT_COLS].to_numpy(dtype=np.float32, copy=False)

    logging.info(f"Final cleaned matrix shape: {X.shape}")

    for method in methods:
        logging.info(f"Scoring method: {method}")

        try:
            if method == "ecod":
                model, scores = fit_and_score_ecod(X)
            elif method == "copod":
                model, scores = fit_and_score_copod(X)
            else:
                logging.warning(f"Unsupported method in config: {method}, skipping.")
                continue
        except Exception as e:
            logging.exception(f"Failed to fit/score method {method}: {e}")
            continue

        n_rows = len(scores)
        if n_rows == 0:
            logging.warning("No rows present after filtering; skipping.")
            continue

        if top_n_threshold >= n_rows:
            threshold_value = float(np.min(scores))
        else:
            threshold_value = float(np.partition(scores, -top_n_threshold)[-top_n_threshold])

        hist_path = hist_dir / f"{parquet_path.stem}_{method}_hist_top{top_n_threshold}.png"
        save_histogram(
            scores=scores,
            threshold_value=threshold_value,
            top_n_threshold=top_n_threshold,
            out_path=hist_path,
            title=f"{parquet_path.stem} — {method.upper()}",
        )

        score_col = f"{method}_score"
        df_scored = df.copy()
        df_scored[score_col] = scores

        idx_topk = topk_indices(scores, top_k)
        df_topk = df_scored.iloc[idx_topk].reset_index(drop=True)

        out_path = out_dir / f"{parquet_path.stem}_{method}_{top_k}.parquet"
        try:
            df_topk.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
            logging.info(f"Wrote {len(df_topk)} rows to {out_path}")
        except Exception as e:
            logging.exception(f"Failed to write parquet {out_path}: {e}")

    logging.info("Finished scoring.")


if __name__ == "__main__":
    main()