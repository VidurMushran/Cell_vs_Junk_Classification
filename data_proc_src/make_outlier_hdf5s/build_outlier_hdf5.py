#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import json
import math
import gc
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader
from cellpose import models
from skimage import measure

from slideutils.utils.frame import Frame
from slideutils.utils import utils

# Keep behavior aligned with the existing pipeline
sys.path.append('/mnt/deepstore/Final_DeepPhenotyping/')
from src.utils.utils import channels_to_bgr, load_model, get_embeddings
from src.representation_learning.data_loader import CustomImageDataset


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False


_SLIDE_DIR_CACHE = {}


BACKGROUND_MORPH_PROPS = [
    'area',
    'area_bbox',
    'area_convex',
    'area_filled',
    'axis_major_length',
    'axis_minor_length',
    'eccentricity',
    'equivalent_diameter_area',
    'euler_number',
    'extent',
    'feret_diameter_max',
    'orientation',
    'perimeter',
    'perimeter_crofton',
    'solidity',
]

ARRAY_PROPS = [
    'bbox',
    'centroid',
    'centroid_local',
    'inertia_tensor',
    'inertia_tensor_eigvals',
    'moments_hu',
]

CELL_EXTRA_PROPS = [
    'area_bbox',
    'area_convex',
    'area_filled',
    'axis_major_length',
    'axis_minor_length',
    'equivalent_diameter_area',
    'euler_number',
    'extent',
    'feret_diameter_max',
    'orientation',
    'perimeter',
    'perimeter_crofton',
    'solidity',
]


def normalize_slide_id(s):
    s = str(s).strip()
    if not s:
        return s
    return s if s.startswith('0') else '0' + s



def generate_tile_paths(path, frame_id, starts, name_format):
    return [f"{path}/{name_format}" % (frame_id + j - 1) for j in starts]



def find_slide_directory(slide_id):
    if slide_id in _SLIDE_DIR_CACHE:
        return _SLIDE_DIR_CACHE[slide_id]

    tube_id = slide_id[:5]
    for d in os.listdir('/mnt'):
        base = os.path.join('/mnt', d)
        if not os.path.isdir(base):
            continue

        pattern = os.path.join(
            base,
            'Oncoscope',
            f'tubeID_{tube_id}',
            '*',
            f'slideID_{slide_id}',
            'bzScanner',
            'proc',
        )
        matches = glob.glob(pattern)
        if not matches:
            continue

        image_dir = matches[0]
        if glob.glob(os.path.join(image_dir, 'Tile*.tif')):
            _SLIDE_DIR_CACHE[slide_id] = (image_dir, 'Tile%06d.tif')
            return _SLIDE_DIR_CACHE[slide_id]
        if glob.glob(os.path.join(image_dir, 'Tile*.jpg')):
            _SLIDE_DIR_CACHE[slide_id] = (image_dir, 'Tile%06d.jpg')
            return _SLIDE_DIR_CACHE[slide_id]

    _SLIDE_DIR_CACHE[slide_id] = (None, None)
    return None, None



def infer_slide_id(parquet_path, df, cfg):
    if cfg.get('slide_id'):
        return normalize_slide_id(cfg['slide_id'])

    if 'slide_id' in df.columns and df['slide_id'].notna().any():
        vals = df['slide_id'].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            return normalize_slide_id(vals[0])

    p = Path(parquet_path)
    stem = p.stem
    if stem:
        token = stem.split('_')[0]
        if token and any(ch.isdigit() for ch in token):
            return normalize_slide_id(token)

    parent = p.parent.name
    if parent and any(ch.isdigit() for ch in parent):
        return normalize_slide_id(parent)

    raise RuntimeError('Could not infer slide_id. Add slide_id to the config or input parquet.')



def read_input_table(path):
    if path.endswith(('.parquet', '.parquet.gz')):
        return pd.read_parquet(path)
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_table(path)



def normalize_input_dataframe(df, slide_id):
    df = df.copy()

    if 'cell_x' in df.columns and 'x' not in df.columns:
        df = df.rename(columns={'cell_x': 'x'})
    if 'cell_y' in df.columns and 'y' not in df.columns:
        df = df.rename(columns={'cell_y': 'y'})

    required = {'frame_id', 'x', 'y'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Missing required columns: {missing}')

    if 'slide_id' not in df.columns:
        df['slide_id'] = slide_id
    else:
        df['slide_id'] = df['slide_id'].fillna(slide_id).astype(str).map(normalize_slide_id)

    if 'cell_id' not in df.columns:
        df['cell_id'] = np.arange(len(df), dtype=np.int64)

    if 'image_id' not in df.columns:
        df['image_id'] = np.arange(len(df), dtype=np.int64)

    if 'source_row_index' not in df.columns:
        df['source_row_index'] = np.arange(len(df), dtype=np.int64)

    df['frame_id'] = df['frame_id'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['x_extract'] = np.rint(df['x']).astype(int)
    df['y_extract'] = np.rint(df['y']).astype(int)
    df['__row_order__'] = np.arange(len(df), dtype=np.int64)
    return df



def resolve_slide_dirs(df, cfg):
    out = {}
    for slide_id in sorted(df['slide_id'].astype(str).unique()):
        if cfg.get('images_path'):
            image_dir = cfg['images_path']
            tile_format = cfg.get('tile_format')
            if not tile_format:
                if os.path.exists(os.path.join(image_dir, 'Tile000001.tif')):
                    tile_format = 'Tile%06d.tif'
                elif os.path.exists(os.path.join(image_dir, 'Tile000001.jpg')):
                    tile_format = 'Tile%06d.jpg'
                else:
                    raise RuntimeError(f'Could not detect tile format in {image_dir}')
            out[slide_id] = (image_dir, tile_format)
        else:
            image_dir, tile_format = find_slide_directory(slide_id)
            if image_dir is None:
                raise RuntimeError(f'Could not find image directory for slide {slide_id}')
            out[slide_id] = (image_dir, tile_format)
    return out



def process_frame_extract(job, slide_dirs, channels, starts, width):
    slide_id, frame_id, frame_df = job
    image_dir, image_format = slide_dirs[slide_id]
    paths = generate_tile_paths(image_dir, frame_id, starts, image_format)

    frame = Frame(frame_id, channels, paths)
    frame.readImage()

    crop_df = pd.DataFrame({
        'x': frame_df['x_extract'].astype(int).values,
        'y': frame_df['y_extract'].astype(int).values,
        'cell_id': frame_df['cell_id'].astype(int).values,
    }, index=frame_df.index)

    images, _ = frame.extract_crops(crop_df, width, mask_flag=False)
    return frame_df['__row_order__'].values, images



def get_center_mask(arr):
    arr = np.asarray(arr).copy()
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    center_value = int(arr[cy, cx])

    if center_value > 0:
        out = (arr == center_value).astype(np.uint16)
        return out

    ys, xs = np.nonzero(arr > 0)
    if len(ys) == 0:
        return np.zeros_like(arr, dtype=np.uint16)

    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    k = int(np.argmin(d2))
    nearest_label = int(arr[ys[k], xs[k]])
    out = (arr == nearest_label).astype(np.uint16)
    return out



def run_cellpose_segmentation(images, cfg):
    device = cfg.get('device', 'cuda:0')
    use_gpu = device != 'cpu'
    cp_model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=cfg['mask_model_path'],
        device=torch.device(device),
    )

    masks = np.zeros((images.shape[0], images.shape[1], images.shape[2], 1), dtype=np.uint16)
    batch_size = int(cfg.get('cellpose_batch_size', 8))
    diameter = float(cfg.get('mask_diameter', 15))

    for start in tqdm.tqdm(range(0, len(images), batch_size), desc='Segmenting crops'):
        stop = min(start + batch_size, len(images))
        rgb_batch = [channels_to_bgr(im, [0, 3], [2, 3], [1, 3]).squeeze(0) for im in images[start:stop]]
        pred_masks, _, _ = cp_model.eval(rgb_batch, diameter=diameter, channels=[0, 0], batch_size=batch_size)
        for i, pm in enumerate(pred_masks):
            masks[start + i, ..., 0] = get_center_mask(pm)

    del cp_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return masks



def flatten_value(prefix, key, value, out):
    if np.isscalar(value):
        if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
            return
        out[f'{prefix}{key}'] = value
        return

    arr = np.asarray(value)
    if arr.ndim == 0:
        val = arr.item()
        if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
            return
        out[f'{prefix}{key}'] = val
        return

    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
            continue
        idx_txt = '_'.join(str(i) for i in idx)
        out[f'{prefix}{key}_{idx_txt}'] = val



def compute_regionprops_dict(label_img, intensity_img=None, prefix=''):
    out = {}
    if np.count_nonzero(label_img) == 0:
        return out

    props = measure.regionprops(label_img.astype(np.uint8), intensity_image=intensity_img)
    if not props:
        return out
    rp = props[0]

    prop_names = BACKGROUND_MORPH_PROPS + ARRAY_PROPS
    if intensity_img is not None:
        prop_names = prop_names + ['intensity_max', 'intensity_mean', 'intensity_min']

    for key in prop_names:
        if not hasattr(rp, key):
            continue
        try:
            value = getattr(rp, key)
            flatten_value(prefix, key, value, out)
        except Exception:
            continue
    return out



def compute_basic_and_regionprops_for_event(args):
    row_order, image, mask, channels = args
    mask2d = mask[..., 0].astype(np.uint8)

    out = {'__row_order__': int(row_order)}

    if np.count_nonzero(mask2d) > 0:
        props = measure.regionprops_table(
            mask2d,
            image,
            separator='_',
            properties=['area', 'eccentricity', 'intensity_mean'],
        )
        if len(props['area']) > 0:
            out['area'] = int(props['area'][0])
            out['eccentricity'] = float(props['eccentricity'][0])
            means = np.asarray(props['intensity_mean'][0]).ravel()
            for ch_name, v in zip(channels, means.tolist()):
                out[f'{ch_name}_mean'] = float(v)

        cell_prop_dict = compute_regionprops_dict(mask2d.astype(np.uint8), intensity_img=None, prefix='cell_prop_')
        out.update(cell_prop_dict)
    else:
        out['area'] = 0
        out['eccentricity'] = np.nan
        for ch_name in channels:
            out[f'{ch_name}_mean'] = np.nan

    background = (mask2d == 0).astype(np.uint8)
    bg_prop_dict = compute_regionprops_dict(background, intensity_img=None, prefix='background_')
    out.update(bg_prop_dict)

    for ch_idx, ch_name in enumerate(channels):
        vals = image[..., ch_idx][background > 0]
        if vals.size == 0:
            out[f'background_{ch_name}_mean'] = np.nan
            out[f'background_{ch_name}_min'] = np.nan
            out[f'background_{ch_name}_max'] = np.nan
            out[f'background_{ch_name}_std'] = np.nan
        else:
            out[f'background_{ch_name}_mean'] = float(vals.mean())
            out[f'background_{ch_name}_min'] = float(vals.min())
            out[f'background_{ch_name}_max'] = float(vals.max())
            out[f'background_{ch_name}_std'] = float(vals.std())

    return out



def compute_features(images, masks, channels, workers):
    jobs = [(i, images[i], masks[i], channels) for i in range(len(images))]
    n_proc = max(1, int(workers))

    if n_proc == 1:
        rows = [compute_basic_and_regionprops_for_event(j) for j in tqdm.tqdm(jobs, desc='Computing regionprops')]
    else:
        with mp.Pool(n_proc) as pool:
            rows = list(tqdm.tqdm(pool.imap(compute_basic_and_regionprops_for_event, jobs), total=len(jobs), desc='Computing regionprops'))

    feat_df = pd.DataFrame(rows).sort_values('__row_order__').reset_index(drop=True)
    return feat_df



def compute_embeddings(images, masks, cfg):
    if not cfg.get('encode_model_path'):
        return None

    device = cfg.get('device', 'cuda:0')
    model = load_model(cfg['encode_model_path'], device=device).to(device).eval()
    labels = np.zeros(images.shape[0], dtype=np.int64)
    dataset = CustomImageDataset(images, masks, labels=labels, tran=False)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.get('embedding_batch_size', 4096)),
        shuffle=False,
        num_workers=max(0, int(cfg.get('embedding_workers', cfg.get('workers', 4)))),
    )

    embeddings = get_embeddings(model, dataloader, device).numpy()
    emb_df = pd.DataFrame(embeddings.astype(np.float16), columns=[f'z{i}' for i in range(embeddings.shape[1])])

    del model, dataloader, dataset, embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return emb_df



def write_hdf5(output_h5, images, masks, channels, features):
    output_h5 = str(output_h5)
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('images', data=images, dtype='uint16')
        f.create_dataset('masks', data=masks, dtype='uint16')
        f.create_dataset('channels', data=np.array(channels, dtype='S'))

    features.to_hdf(output_h5, key='features', mode='a', format='table', data_columns=['image_id', 'slide_id', 'frame_id'])



def build_jobs(df):
    jobs = []
    for (slide_id, frame_id), g in df.groupby(['slide_id', 'frame_id'], sort=True):
        jobs.append((slide_id, int(frame_id), g.copy()))
    return jobs



def extract_all_crops(df, slide_dirs, cfg):
    n = len(df)
    width = int(cfg.get('width', 75))
    channels = cfg['channels']
    starts = cfg['starts']
    jobs = build_jobs(df)

    images = np.zeros((n, width, width, len(channels)), dtype=np.uint16)
    worker_fn = partial(
        process_frame_extract,
        slide_dirs=slide_dirs,
        channels=channels,
        starts=starts,
        width=width,
    )

    n_proc = max(1, int(cfg.get('workers', 4)))
    if n_proc == 1:
        iterator = map(worker_fn, jobs)
        for row_orders, batch_images in tqdm.tqdm(iterator, total=len(jobs), desc='Extracting crops'):
            images[row_orders] = batch_images
    else:
        with mp.Pool(n_proc) as pool:
            iterator = pool.imap(worker_fn, jobs)
            for row_orders, batch_images in tqdm.tqdm(iterator, total=len(jobs), desc='Extracting crops'):
                images[row_orders] = batch_images
    return images



def reorder_columns(features):
    preferred = ['image_id', 'slide_id', 'frame_id', 'cell_id', 'x', 'y', 'area', 'eccentricity', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']
    ordered = [c for c in preferred if c in features.columns]
    rest = [c for c in features.columns if c not in ordered]
    return features[ordered + rest]



def main():
    ap = argparse.ArgumentParser(description='Rebuild an HDF5 with images, masks, features, and BLUE embeddings from BLUE or RED-BLUE parquet outputs.')
    ap.add_argument('--config', required=True, help='Path to YAML config file')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if not cfg.get('input_parquet'):
        raise RuntimeError('Config must include input_parquet')
    if not cfg.get('output_h5'):
        raise RuntimeError('Config must include output_h5')
    if not cfg.get('mask_model_path'):
        raise RuntimeError('Config must include mask_model_path')

    logger = utils.get_logger(__name__, cfg.get('verbose', 0))

    df = read_input_table(cfg['input_parquet'])
    slide_id = infer_slide_id(cfg['input_parquet'], df, cfg)
    df = normalize_input_dataframe(df, slide_id)
    slide_dirs = resolve_slide_dirs(df, cfg)

    logger.info(f'Read {len(df):,} rows from {cfg["input_parquet"]}')
    logger.info(f'Found {len(slide_dirs)} slide(s): {", ".join(sorted(slide_dirs.keys()))}')

    images = extract_all_crops(df, slide_dirs, cfg)
    masks = run_cellpose_segmentation(images, cfg)
    props_df = compute_features(images, masks, cfg['channels'], cfg.get('feature_workers', cfg.get('workers', 4)))

    features = df.sort_values('__row_order__').reset_index(drop=True).copy()
    props_df = props_df.sort_values('__row_order__').reset_index(drop=True)
    features = features.merge(props_df, on='__row_order__', how='left')

    emb_df = compute_embeddings(images, masks, cfg)
    if emb_df is not None:
        for c in emb_df.columns:
            features[c] = emb_df[c].values

    drop_cols = [c for c in ['x_extract', 'y_extract', '__row_order__'] if c in features.columns]
    features = features.drop(columns=drop_cols)
    features = reorder_columns(features)

    write_hdf5(cfg['output_h5'], images, masks, cfg['channels'], features)

    logger.info(f'Wrote HDF5: {cfg["output_h5"]}')
    logger.info(f'images shape: {images.shape}')
    logger.info(f'masks shape: {masks.shape}')
    logger.info(f'features rows: {len(features):,}')


if __name__ == '__main__':
    main()
