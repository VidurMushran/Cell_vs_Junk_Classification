#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score full test HDF5s with one or more trained models and append
model_score_{model_name} columns to /features/table.

Supports:
- sklearn / xgboost / lightgbm style pickle/joblib models using tabular features
- TorchScript models
- PyTorch checkpoints with an importable model class

Inputs available per event:
- image: HDF5 dataset /images, shape (N, H, W, C)
- mask:  HDF5 dataset /masks,  shape (N, H, W)
- features: pandas table /features/table

Typical usage:
python score_full_hdf5s.py \
    --manifest model_manifest.yml \
    --input-glob "test_data/*.full.hdf5" \
    --batch-size 256 \
    --device cuda3

The manifest controls:
- model name
- model type
- file path
- required inputs: image, mask, features
- feature columns
- optional feature scaler
- optional class index
- optional PyTorch model import path
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ----------------------------
# Utilities
# ----------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dynamic_import(path: str):
    """
    Import object from a string like:
    mypackage.mymodule:MyClass
    """
    module_name, obj_name = path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if TORCH_AVAILABLE and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_4d_image(img: np.ndarray) -> np.ndarray:
    """
    Input HDF5 images are assumed (B, H, W, C). Convert to torch style (B, C, H, W).
    """
    if img.ndim != 4:
        raise ValueError(f"Expected image batch to be 4D, got shape {img.shape}")
    return np.transpose(img, (0, 3, 1, 2))


def ensure_4d_mask(mask: np.ndarray) -> np.ndarray:
    """
    Input HDF5 masks are assumed (B, H, W). Convert to (B, 1, H, W).
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected mask batch to be 3D, got shape {mask.shape}")
    return mask[:, None, :, :]


def normalize_uint_image(x: np.ndarray, mode: str = "255") -> np.ndarray:
    """
    Normalize images into float32.
    Modes:
    - 255: divide by 255
    - 65535: divide by 65535
    - none: cast only
    """
    x = x.astype(np.float32, copy=False)
    if mode == "255":
        x /= 255.0
    elif mode == "65535":
        x /= 65535.0
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown image normalization mode: {mode}")
    return x


def mask_to_float(mask: np.ndarray, binarize: bool = True) -> np.ndarray:
    mask = mask.astype(np.float32, copy=False)
    if binarize:
        mask = (mask > 0).astype(np.float32)
    return mask


def find_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ----------------------------
# Model wrappers
# ----------------------------

class BaseScorer:
    def predict_scores(
        self,
        image: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        features: Optional[pd.DataFrame],
    ) -> np.ndarray:
        raise NotImplementedError


class SklearnFeatureScorer(BaseScorer):
    def __init__(
        self,
        model_path: str,
        feature_columns: Sequence[str],
        positive_class_index: int = 1,
        scaler_path: Optional[str] = None,
    ):
        self.model = joblib.load(model_path)
        self.feature_columns = list(feature_columns)
        self.positive_class_index = positive_class_index
        self.scaler = joblib.load(scaler_path) if scaler_path else None

    def predict_scores(self, image, mask, features) -> np.ndarray:
        if features is None:
            raise ValueError("features are required for this model")
        X = features[self.feature_columns].to_numpy(dtype=np.float32, copy=False)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.ndim == 2:
                return proba[:, self.positive_class_index].astype(np.float32)
            return np.asarray(proba, dtype=np.float32)

        if hasattr(self.model, "decision_function"):
            dec = self.model.decision_function(X)
            dec = np.asarray(dec)
            if dec.ndim == 1:
                return sigmoid(dec).astype(np.float32)
            probs = softmax(dec, axis=1)
            return probs[:, self.positive_class_index].astype(np.float32)

        preds = self.model.predict(X)
        return np.asarray(preds, dtype=np.float32)


class TorchScorer(BaseScorer):
    def __init__(
        self,
        model: "nn.Module",
        input_mode: str,
        device: str = "cpu",
        feature_columns: Optional[Sequence[str]] = None,
        scaler_path: Optional[str] = None,
        positive_class_index: int = 1,
        image_norm: str = "255",
        mask_binarize: bool = True,
        output_activation: str = "auto",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.input_mode = input_mode
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.positive_class_index = positive_class_index
        self.image_norm = image_norm
        self.mask_binarize = mask_binarize
        self.output_activation = output_activation

    def _prepare_inputs(
        self,
        image: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        features: Optional[pd.DataFrame],
    ):
        img_t = None
        mask_t = None
        feat_t = None

        if image is not None:
            image = normalize_uint_image(image, self.image_norm)
            image = ensure_4d_image(image)
            img_t = torch.from_numpy(image).to(self.device)

        if mask is not None:
            mask = mask_to_float(mask, binarize=self.mask_binarize)
            mask = ensure_4d_mask(mask)
            mask_t = torch.from_numpy(mask).to(self.device)

        if self.feature_columns is not None:
            if features is None:
                raise ValueError("This model requires tabular features")
            X = features[self.feature_columns].to_numpy(dtype=np.float32, copy=False)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            feat_t = torch.from_numpy(X).to(self.device)

        return img_t, mask_t, feat_t

    def _postprocess_output(self, out) -> np.ndarray:
        out = as_numpy(out)

        if out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]

        if self.output_activation == "sigmoid":
            return sigmoid(out).astype(np.float32)

        if self.output_activation == "softmax":
            if out.ndim == 1:
                return sigmoid(out).astype(np.float32)
            return softmax(out, axis=1)[:, self.positive_class_index].astype(np.float32)

        # auto
        if out.ndim == 1:
            if np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0:
                return out.astype(np.float32)
            return sigmoid(out).astype(np.float32)

        if out.ndim == 2:
            if out.shape[1] == 1:
                x = out[:, 0]
                if np.nanmin(x) >= 0.0 and np.nanmax(x) <= 1.0:
                    return x.astype(np.float32)
                return sigmoid(x).astype(np.float32)
            row_sums = out.sum(axis=1)
            if np.allclose(row_sums, 1.0, atol=1e-3):
                return out[:, self.positive_class_index].astype(np.float32)
            probs = softmax(out, axis=1)
            return probs[:, self.positive_class_index].astype(np.float32)

        raise ValueError(f"Unexpected model output shape: {out.shape}")

    @torch.no_grad()
    def predict_scores(self, image, mask, features) -> np.ndarray:
        img_t, mask_t, feat_t = self._prepare_inputs(image, mask, features)

        if self.input_mode == "features":
            out = self.model(feat_t)
        elif self.input_mode == "image":
            out = self.model(img_t)
        elif self.input_mode == "image_mask":
            try:
                out = self.model(img_t, mask_t)
            except TypeError:
                concat = torch.cat([img_t, mask_t], dim=1)
                out = self.model(concat)
        elif self.input_mode == "image_features":
            out = self.model(img_t, feat_t)
        elif self.input_mode == "image_mask_features":
            try:
                out = self.model(img_t, mask_t, feat_t)
            except TypeError:
                concat = torch.cat([img_t, mask_t], dim=1)
                out = self.model(concat, feat_t)
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        return self._postprocess_output(out)


def load_torch_model_from_manifest(spec: Dict[str, Any], device: str) -> TorchScorer:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")

    model_path = spec["path"]
    model_format = spec.get("format", "checkpoint")

    if model_format == "torchscript":
        model = torch.jit.load(model_path, map_location=device)
    else:
        model_class = dynamic_import(spec["model_class"])
        model_kwargs = spec.get("model_kwargs", {})
        model = model_class(**model_kwargs)

        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=spec.get("strict", True))

    return TorchScorer(
        model=model,
        input_mode=spec["input_mode"],
        device=device,
        feature_columns=spec.get("feature_columns"),
        scaler_path=spec.get("scaler_path"),
        positive_class_index=spec.get("positive_class_index", 1),
        image_norm=spec.get("image_norm", "255"),
        mask_binarize=spec.get("mask_binarize", True),
        output_activation=spec.get("output_activation", "auto"),
    )


def load_scorer(spec: Dict[str, Any], device: str) -> BaseScorer:
    model_type = spec["type"]

    if model_type == "sklearn":
        return SklearnFeatureScorer(
            model_path=spec["path"],
            feature_columns=spec["feature_columns"],
            positive_class_index=spec.get("positive_class_index", 1),
            scaler_path=spec.get("scaler_path"),
        )

    if model_type == "torch":
        return load_torch_model_from_manifest(spec, device=device)

    raise ValueError(f"Unsupported model type: {model_type}")


# ----------------------------
# HDF5 I/O
# ----------------------------

def read_features_df(h5_path: str) -> pd.DataFrame:
    return pd.read_hdf(h5_path, key="/features/table")


def write_features_df(h5_path: str, df: pd.DataFrame) -> None:
    with pd.HDFStore(h5_path, mode="a") as store:
        if "/features/table" in store:
            del store["/features/table"]
        store.put("/features/table", df, format="table", data_columns=True)


def get_num_events(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        return int(f["images"].shape[0])


def read_h5_batch(
    h5_path: str,
    start: int,
    stop: int,
    need_image: bool,
    need_mask: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    with h5py.File(h5_path, "r") as f:
        image = f["images"][start:stop] if need_image else None
        mask = f["masks"][start:stop] if need_mask and "masks" in f else None
    return image, mask


# ----------------------------
# Scoring
# ----------------------------

@dataclass
class ModelJob:
    name: str
    scorer: BaseScorer
    need_image: bool
    need_mask: bool
    need_features: bool


def score_one_model_on_hdf5(
    h5_path: str,
    features_df: pd.DataFrame,
    job: ModelJob,
    batch_size: int,
) -> np.ndarray:
    n = len(features_df)
    scores = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch_features = features_df.iloc[start:stop] if job.need_features else None
        batch_image, batch_mask = read_h5_batch(
            h5_path,
            start,
            stop,
            need_image=job.need_image,
            need_mask=job.need_mask,
        )
        batch_scores = job.scorer.predict_scores(batch_image, batch_mask, batch_features)
        batch_scores = np.asarray(batch_scores, dtype=np.float32).reshape(-1)

        if len(batch_scores) != (stop - start):
            raise ValueError(
                f"Model {job.name} returned {len(batch_scores)} scores "
                f"for batch size {stop - start}"
            )

        scores[start:stop] = batch_scores

    return scores


def build_jobs_from_manifest(manifest: Dict[str, Any], device: str) -> List[ModelJob]:
    jobs = []
    for spec in manifest["models"]:
        scorer = load_scorer(spec, device=device)
        input_mode = spec["input_mode"]

        jobs.append(
            ModelJob(
                name=spec["name"],
                scorer=scorer,
                need_image=input_mode in {"image", "image_mask", "image_features", "image_mask_features"},
                need_mask=input_mode in {"image_mask", "image_mask_features"},
                need_features=input_mode in {"features", "image_features", "image_mask_features"},
            )
        )
    return jobs


def score_hdf5_file(
    h5_path: str,
    jobs: List[ModelJob],
    batch_size: int,
    overwrite: bool = False,
) -> None:
    print(f"\nScoring: {h5_path}")
    features_df = read_features_df(h5_path)
    n = len(features_df)
    print(f"  rows in /features/table: {n}")

    total_n = get_num_events(h5_path)
    if total_n != n:
        raise ValueError(
            f"Mismatch between images ({total_n}) and feature rows ({n}) in {h5_path}"
        )

    for job in jobs:
        col = f"model_score_{job.name}"
        if (col in features_df.columns) and not overwrite:
            print(f"  skipping {job.name} because {col} already exists")
            continue

        print(f"  running model: {job.name}")
        scores = score_one_model_on_hdf5(
            h5_path=h5_path,
            features_df=features_df,
            job=job,
            batch_size=batch_size,
        )
        features_df[col] = scores
        print(
            f"    wrote scores -> {col} "
            f"(min={np.nanmin(scores):.4f}, max={np.nanmax(scores):.4f}, mean={np.nanmean(scores):.4f})"
        )

    write_features_df(h5_path, features_df)
    print(f"  saved updated features table to {h5_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True, help="YAML manifest describing models")
    p.add_argument("--input-glob", type=str, required=True, help='Glob for full test HDF5s, e.g. "test_data/*.full.hdf5"')
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    h5_paths = sorted(glob(args.input_glob))

    if len(h5_paths) == 0:
        raise FileNotFoundError(f"No HDF5 files matched: {args.input_glob}")

    jobs = build_jobs_from_manifest(manifest, device=args.device)

    print("Models loaded:")
    for job in jobs:
        print(
            f"  - {job.name}: "
            f"image={job.need_image}, mask={job.need_mask}, features={job.need_features}"
        )

    for h5_path in h5_paths:
        score_hdf5_file(
            h5_path=h5_path,
            jobs=jobs,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()