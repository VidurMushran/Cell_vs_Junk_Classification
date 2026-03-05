#!/usr/bin/env python3
import sys
import os
import gc
import math
import json
import argparse
import multiprocessing as mp
from functools import partial

import yaml
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
from skimage import measure
from torch.utils.data import DataLoader, Dataset
from cellpose import models

from pyod.models.copod import COPOD
from sklearn.preprocessing import StandardScaler

# Keep imports aligned with the existing codebase as much as possible
sys.path.append('/mnt/deepstore/Final_DeepPhenotyping/')
from src.utils.utils import channels_to_bgr, load_model, get_embeddings  # noqa: E402
from src.representation_learning.data_loader import CustomImageDataset  # noqa: E402

try:
    from src.leukocyte_classifier.wbc_classifier import CNNModel  # noqa: E402
    WBC_AVAILABLE = True
except Exception:
    CNNModel = None
    WBC_AVAILABLE = False


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

CP_MODEL = None
CP_MODEL_ARGS = None

CELL_BASIC_DTYPES = {
    'frame_id': 'uint32',
    'cell_id': 'uint32',
    'y': 'float32',
    'x': 'float32',
    'area': 'uint32',
    'eccentricity': 'float32',
    'DAPI_mean': 'float32',
    'TRITC_mean': 'float32',
    'CY5_mean': 'float32',
    'FITC_mean': 'float32',
}

BACKGROUND_REGIONPROP_EXCLUDE = {
    'coords', 'coords_scaled', 'image', 'image_convex', 'image_filled',
    'image_intensity', 'slice', 'label', 'num_pixels'
}


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def normalize_device(device):
    if device is None:
        return 'cpu'
    return str(device)


def get_center_mask(arr):
    """
    Keep only the component with the same label as the center pixel.
    This intentionally mirrors the existing helper logic.
    """
    arr = np.asarray(arr).copy()
    center_x = arr.shape[0] // 2
    center_y = arr.shape[1] // 2
    center_value = arr[center_x, center_y]

    if center_value == 0:
        arr[...] = 0
        return arr.astype(np.uint16)

    mask = (arr == center_value)
    arr[~mask] = 0
    arr[mask] = 1
    return arr.astype(np.uint16)


def init_segment_worker(model_path, device):
    global CP_MODEL, CP_MODEL_ARGS
    CP_MODEL_ARGS = (model_path, device)
    CP_MODEL = models.CellposeModel(
        gpu=(str(device) != 'cpu'),
        pretrained_model=model_path,
        device=torch.device(device),
    )


def _segment_images_with_global_model(images, diameter, batch_size):
    global CP_MODEL
    out = np.zeros(images.shape[:3], dtype=np.uint16)

    for i, image in enumerate(images):
        rgb = channels_to_bgr(image, [0, 3], [2, 3], [1, 3])
        mask, _, _ = CP_MODEL.eval(
            rgb,
            diameter=diameter,
            channels=[0, 0],
            batch_size=batch_size,
        )
        out[i] = get_center_mask(mask)

    return out


def segment_chunk(job):
    input_h5, start, end, dataset_key, diameter, batch_size = job
    with h5py.File(input_h5, 'r') as f:
        images = f[dataset_key][start:end]
    masks = _segment_images_with_global_model(images, diameter=diameter, batch_size=batch_size)
    return start, end, masks


def segment_all_images(input_h5, image_dataset_key, n_images, cfg):
    workers = int(cfg.get('workers', 1))
    diameter = int(cfg.get('cellpose_diameter', 20))
    batch_size = int(cfg.get('cellpose_batch_size', 8))
    chunk_size = int(cfg.get('mask_chunk_size', 256))
    model_path = cfg['mask_model_path']
    device = normalize_device(cfg.get('device', 'cpu'))

    jobs = []
    for start in range(0, n_images, chunk_size):
        end = min(start + chunk_size, n_images)
        jobs.append((input_h5, start, end, image_dataset_key, diameter, batch_size))

    masks = np.zeros((n_images, 75, 75), dtype=np.uint16)

    # Multiprocessing with GPU is possible but can be memory-heavy because each worker loads the model.
    # We still allow it for consistency with the user's request.
    ctx = mp.get_context('spawn')
    with ctx.Pool(
        processes=workers,
        initializer=init_segment_worker,
        initargs=(model_path, device),
    ) as pool:
        for start, end, chunk_masks in tqdm.tqdm(
            pool.imap_unordered(segment_chunk, jobs),
            total=len(jobs),
            desc='Segmenting crops',
        ):
            masks[start:end] = chunk_masks

    return masks


def _flatten_value(prefix, value, out):
    if np.isscalar(value) or isinstance(value, (int, float, np.number, bool)):
        out[prefix] = value
        return

    value = np.asarray(value)
    if value.dtype == object:
        return
    if value.size == 0:
        return
    if value.size > 256:
        return

    for idx in np.ndindex(value.shape):
        flat_key = prefix + '_' + '_'.join(str(i) for i in idx)
        out[flat_key] = value[idx].item() if hasattr(value[idx], 'item') else value[idx]


def extract_background_regionprops(image, mask):
    bg = (mask == 0).astype(np.uint8)
    labeled_bg = measure.label(bg, connectivity=1)
    props = measure.regionprops(labeled_bg, intensity_image=image)

    out = {}
    if len(props) == 0:
        return out

    region = max(props, key=lambda r: r.area)

    for attr in dir(region):
        if attr.startswith('_'):
            continue
        if attr in BACKGROUND_REGIONPROP_EXCLUDE:
            continue

        try:
            value = getattr(region, attr)
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(value, tuple):
            value = np.asarray(value)

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.dtype == object:
                continue
            if arr.size > 256:
                continue
            _flatten_value('background_' + attr, arr, out)
        elif np.isscalar(value) or isinstance(value, (int, float, np.number, bool)):
            out['background_' + attr] = value

    return out


def calc_basic_features_single(image, mask, channels):
    label_img = mask.astype(np.uint16)
    if label_img.max() == 0:
        return {
            'area': 0,
            'eccentricity': np.nan,
            **{f'{ch}_mean': 0.0 for ch in channels},
        }

    props = measure.regionprops_table(
        label_img,
        image,
        separator='_',
        properties=['label', 'area', 'eccentricity', 'intensity_mean'],
    )
    props = pd.DataFrame(props)
    if len(props) == 0:
        return {
            'area': 0,
            'eccentricity': np.nan,
            **{f'{ch}_mean': 0.0 for ch in channels},
        }

    row = props.iloc[0]
    out = {
        'area': int(row['area']),
        'eccentricity': float(row['eccentricity']) if pd.notnull(row['eccentricity']) else np.nan,
    }

    for i, ch in enumerate(channels):
        col = f'intensity_mean_{i}'
        out[f'{ch}_mean'] = float(row[col]) if col in row else 0.0

    return out


def feature_chunk(job):
    input_h5, start, end, dataset_key, masks, channels = job
    rows = []
    with h5py.File(input_h5, 'r') as f:
        images = f[dataset_key][start:end]

    for i in range(images.shape[0]):
        image = images[i]
        mask = masks[i]
        basic = calc_basic_features_single(image, mask, channels)
        bg = extract_background_regionprops(image, mask)
        merged = {}
        merged.update(basic)
        merged.update(bg)
        rows.append(merged)

    return start, end, rows


def extract_all_features(input_h5, image_dataset_key, masks, channels, cfg):
    workers = int(cfg.get('feature_workers', cfg.get('workers', 1)))
    chunk_size = int(cfg.get('feature_chunk_size', 512))
    n_images = masks.shape[0]

    jobs = []
    for start in range(0, n_images, chunk_size):
        end = min(start + chunk_size, n_images)
        jobs.append((input_h5, start, end, image_dataset_key, masks[start:end], channels))

    rows = [None] * len(jobs)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=workers) as pool:
        for start, end, chunk_rows in tqdm.tqdm(
            pool.imap_unordered(feature_chunk, jobs),
            total=len(jobs),
            desc='Calculating basic/background features',
        ):
            rows[start // chunk_size] = chunk_rows

    flat = []
    for chunk_rows in rows:
        if chunk_rows is not None:
            flat.extend(chunk_rows)

    return pd.DataFrame(flat)


class H5ImageMaskDataset(Dataset):
    def __init__(self, input_h5, image_dataset_key='images', mask_array=None, labels=None):
        self.input_h5 = input_h5
        self.image_dataset_key = image_dataset_key
        self.mask_array = mask_array
        if labels is None:
            labels = np.zeros(len(mask_array), dtype=np.int64)
        self.labels = labels
        self._hf = None

    def __len__(self):
        return len(self.mask_array)

    def __getitem__(self, idx):
        if self._hf is None:
            self._hf = h5py.File(self.input_h5, 'r')
        image = self._hf[self.image_dataset_key][idx]
        mask = self.mask_array[idx]
        sample = np.concatenate([image, mask[..., np.newaxis]], axis=-1)
        sample = np.transpose(sample, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(sample), self.labels[idx]


def infer_embeddings(input_h5, image_dataset_key, masks, cfg):
    device = normalize_device(cfg.get('device', 'cpu'))
    batch_size = int(cfg.get('inference_batch', 10000))
    num_workers = int(cfg.get('dataloader_workers', 0))

    enc_model = load_model(cfg['encode_model_path'], device=device).to(device).eval()

    dataset = H5ImageMaskDataset(
        input_h5=input_h5,
        image_dataset_key=image_dataset_key,
        mask_array=masks.astype(np.uint16),
        labels=np.zeros(masks.shape[0], dtype=np.int64),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu'),
    )

    embeddings = get_embeddings(enc_model, dataloader, device)
    embeddings = embeddings.numpy().astype(np.float16)
    cols = [f'z{i}' for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=cols)


def infer_wbc_scores(input_h5, image_dataset_key, masks, cfg):
    if not WBC_AVAILABLE:
        raise RuntimeError('WBC classifier imports are not available in this environment.')

    device = normalize_device(cfg.get('device', 'cpu'))
    batch_size = int(cfg.get('wbc_batch_size', 5000))
    num_workers = int(cfg.get('dataloader_workers', 0))

    model = CNNModel()
    state = torch.load(cfg['classifier_path'], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    dataset = H5ImageMaskDataset(
        input_h5=input_h5,
        image_dataset_key=image_dataset_key,
        mask_array=masks.astype(np.uint16),
        labels=np.zeros(masks.shape[0], dtype=np.int64),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu'),
    )

    preds = []
    with torch.no_grad():
        for inputs, _ in tqdm.tqdm(dataloader, desc='Inferring WBC scores'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds.append(probs)

    return np.concatenate(preds)


def maybe_run_outlier_detection(features_df, cfg):
    if not cfg.get('enable_outlier_detection', False):
        return features_df

    z_cols = [c for c in features_df.columns if c.startswith('z')]
    if len(z_cols) == 0:
        return features_df

    feats = features_df[z_cols].astype(np.float64)
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    copod = COPOD(
        contamination=float(cfg.get('outlier_contamination', 0.001)),
        n_jobs=int(cfg.get('workers', 1)),
    )
    copod.fit(feats)
    features_df['outlier_score'] = copod.decision_scores_
    return features_df


def resolve_xy_columns(df):
    df = df.copy()
    if 'x' not in df.columns and 'cell_x' in df.columns:
        df['x'] = df['cell_x']
    if 'y' not in df.columns and 'cell_y' in df.columns:
        df['y'] = df['cell_y']
    if 'cell_x' not in df.columns and 'x' in df.columns:
        df['cell_x'] = df['x']
    if 'cell_y' not in df.columns and 'y' in df.columns:
        df['cell_y'] = df['y']
    return df


def validate_input(metadata_df, images_shape):
    required_any = [('x', 'cell_x'), ('y', 'cell_y')]
    required_exact = ['slide_id', 'frame_id']

    for c in required_exact:
        if c not in metadata_df.columns:
            raise ValueError(f'Missing required metadata column: {c}')

    for a, b in required_any:
        if a not in metadata_df.columns and b not in metadata_df.columns:
            raise ValueError(f'Missing required metadata column: one of {a} or {b}')

    if len(metadata_df) != images_shape[0]:
        raise ValueError(
            f'features table has {len(metadata_df)} rows but images dataset has {images_shape[0]} rows'
        )


def copy_input_to_output(input_h5, output_h5, image_dataset_key='images', channels_dataset_key='channels'):
    with h5py.File(input_h5, 'r') as src, h5py.File(output_h5, 'w') as dst:
        src.copy(image_dataset_key, dst, name=image_dataset_key)
        if channels_dataset_key in src:
            src.copy(channels_dataset_key, dst, name=channels_dataset_key)


def write_output(output_h5, masks, features_df, input_h5, cfg):
    image_dataset_key = cfg.get('image_dataset_key', 'images')
    channels_dataset_key = cfg.get('channels_dataset_key', 'channels')

    copy_input_to_output(input_h5, output_h5, image_dataset_key=image_dataset_key, channels_dataset_key=channels_dataset_key)

    with h5py.File(output_h5, 'a') as f:
        if 'masks' in f:
            del f['masks']
        f.create_dataset(
            'masks',
            data=masks.astype(np.uint16),
            compression=cfg.get('hdf5_compression', 'gzip'),
            compression_opts=int(cfg.get('hdf5_compression_opts', 4)),
        )

    features_df.to_hdf(output_h5, key='features', mode='a', format='table', data_columns=True)


def main():
    parser = argparse.ArgumentParser(
        description='Add Cellpose center masks and BLUE features to an images-only HDF5.'
    )
    parser.add_argument('--config', required=True, help='Path to mask_and_BLUE_config.yml')
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_h5 = cfg['input_h5']
    output_h5 = cfg['output_h5']
    image_dataset_key = cfg.get('image_dataset_key', 'images')
    channels_dataset_key = cfg.get('channels_dataset_key', 'channels')
    channels = cfg.get('channels', ['DAPI', 'TRITC', 'CY5', 'FITC'])

    if len(channels) != 4:
        raise ValueError('This script expects 4 image channels, matching the existing pipeline assumptions.')

    with h5py.File(input_h5, 'r') as f:
        if image_dataset_key not in f:
            raise ValueError(f'HDF5 file does not contain dataset: {image_dataset_key}')
        images_shape = f[image_dataset_key].shape
        if len(images_shape) != 4 or images_shape[1:] != (75, 75, 4):
            raise ValueError(
                f'Expected images dataset shape (N, 75, 75, 4), got {images_shape}'
            )
        if channels_dataset_key in f:
            stored_channels = [
                c.decode() if isinstance(c, bytes) else str(c)
                for c in f[channels_dataset_key][:]
            ]
            if len(stored_channels) == 4:
                channels = stored_channels

    if os.path.abspath(input_h5) == os.path.abspath(output_h5):
        raise ValueError('input_h5 and output_h5 must be different paths. Write to a new output file.')

    metadata_df = pd.read_hdf(input_h5, key='features')
    metadata_df = resolve_xy_columns(metadata_df)
    validate_input(metadata_df, images_shape)

    print(f'Loaded {len(metadata_df):,} events from {input_h5}')
    print('Generating masks...')
    masks = segment_all_images(input_h5, image_dataset_key, len(metadata_df), cfg)

    print('Calculating basic and background features...')
    feature_df = extract_all_features(input_h5, image_dataset_key, masks, channels, cfg)

    if len(feature_df) != len(metadata_df):
        raise RuntimeError(
            f'Feature extraction returned {len(feature_df)} rows for {len(metadata_df)} input rows.'
        )

    # Preserve and standardize metadata columns.
    out_df = metadata_df.copy().reset_index(drop=True)
    if 'image_id' not in out_df.columns:
        out_df.insert(0, 'image_id', np.arange(len(out_df), dtype=np.int64))
    if 'cell_id' not in out_df.columns:
        out_df['cell_id'] = np.arange(len(out_df), dtype=np.int64)

    # Overwrite / append mask-derived features.
    for col in feature_df.columns:
        out_df[col] = feature_df[col].values

    # Infer BLUE features.
    if cfg.get('encode_model_path'):
        print('Inferring BLUE embeddings...')
        embed_df = infer_embeddings(input_h5, image_dataset_key, masks, cfg)
        out_df = pd.concat([out_df, embed_df], axis=1)

    # Optional WBC classifier.
    if cfg.get('enable_wbc', False) and cfg.get('classifier_path'):
        print('Inferring WBC classifier scores...')
        out_df['wbcclass'] = infer_wbc_scores(input_h5, image_dataset_key, masks, cfg)

    out_df = maybe_run_outlier_detection(out_df, cfg)

    # Cast common columns when present.
    for col, dtype in CELL_BASIC_DTYPES.items():
        if col in out_df.columns:
            try:
                out_df[col] = out_df[col].astype(dtype)
            except Exception:
                pass

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    if os.path.exists(output_h5):
        os.remove(output_h5)

    print('Writing output HDF5...')
    write_output(output_h5, masks, out_df, input_h5, cfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'Done. Wrote: {output_h5}')


if __name__ == '__main__':
    main()
