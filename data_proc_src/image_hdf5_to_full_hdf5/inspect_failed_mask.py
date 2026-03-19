#!/usr/bin/env python3
"""
inspect_failed_mask_fixed.py

Improved inspector:
 - saves tile visualizations in a long horizontal row: <outdir>/<slide>_f<frame>_c<cell>_tile_row.png
 - saves crop visualizations separately (225x225 with 75x75 box + HDF5 vs tile mask): <outdir>/<slide>_f<frame>_c<cell>_crop225.png
 - fixes mask alignment: we crop the tile_mask to 225x225, overlay that on the 225 crop, and draw the 75x75 rectangle at the correct position.

Usage example:
python inspect_failed_mask_fixed.py \
  --slide_id 0B22404 \
  --frame_id 1088 \
  --cell_id 1234 \
  --cell_x 120.5 \
  --cell_y 96.7 \
  --input-h5 /path/to/input.hdf5 \
  --cellpose-model /path/to/cellpose_model \
  --outdir /tmp/inspect_out \
  --device cuda:0
"""
import os
import argparse
import glob
import cv2
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import measure
from cellpose import models
import torch
import sys

# optional helper from your repo - fall back if absent
sys.path.append('/mnt/deepstore/Final_DeepPhenotyping/')
try:
    from src.utils.utils import channels_to_bgr
except Exception:
    def channels_to_bgr(arr, a_map, b_map, c_map):
        # Simple fallback mapping: produce a 3-channel image from 4-channel input.
        # This fallback chooses (R=ch2, G=ch1, B=ch0) which matches many lab conventions.
        if arr.ndim == 3 and arr.shape[2] >= 3:
            r = arr[..., 2]
            g = arr[..., 1]
            b = arr[..., 0]
            merged = cv2.merge([b, g, r])
            return merged
        # if something else, just repeat single channel
        ch = arr[..., 0] if arr.ndim == 3 else arr
        ch8 = (normalize_display_channel(ch))
        return cv2.merge([ch8, ch8, ch8])

# ---------- utils for tile discovery (copied/adapted from your pipeline) ----------
_SLIDE_DIR_CACHE = {}

def find_slide_directory(slide_id):
    if slide_id in _SLIDE_DIR_CACHE:
        return _SLIDE_DIR_CACHE[slide_id]

    tube_id = str(slide_id)[:5]
    for d in os.listdir("/mnt"):
        base = os.path.join("/mnt", d)
        if not os.path.isdir(base):
            continue
        pattern = os.path.join(
            base,
            "Oncoscope",
            f"tubeID_{tube_id}",
            "*",
            f"slideID_{slide_id}",
            "bzScanner",
            "proc",
        )
        matches = glob.glob(pattern)
        if not matches:
            continue
        image_dir = matches[0]
        if glob.glob(os.path.join(image_dir, "Tile*.tif")):
            _SLIDE_DIR_CACHE[slide_id] = (image_dir, "Tile%06d.tif")
            return _SLIDE_DIR_CACHE[slide_id]
        if glob.glob(os.path.join(image_dir, "Tile*.jpg")):
            _SLIDE_DIR_CACHE[slide_id] = (image_dir, "Tile%06d.jpg")
            return _SLIDE_DIR_CACHE[slide_id]

    _SLIDE_DIR_CACHE[slide_id] = (None, None)
    return None, None

def generate_tile_paths(path, frame_id, starts, name_format):
    return [f"{path}/{name_format}" % (int(frame_id) + int(j) - 1) for j in starts]

def read_full_tile(slide_id, frame_id, starts):
    image_dir, name_fmt = find_slide_directory(str(slide_id))
    if image_dir is None:
        return None
    paths = generate_tile_paths(image_dir, int(frame_id), starts, name_fmt)
    for p in paths:
        if not os.path.exists(p):
            return None
    images = [cv2.imread(p, -1) for p in paths]  # preserve bit depth
    if any(im is None for im in images):
        return None
    tile = cv2.merge(images)
    return tile  # shape (H, W, C)

# ---------- display helpers ----------
def normalize_display_channel(ch):
    """Stretch to uint8 for display (1-99 percentile)"""
    ch = np.asarray(ch)
    if ch.dtype == np.uint8:
        return ch
    chf = ch.astype(np.float32)
    lo = np.percentile(chf, 1)
    hi = np.percentile(chf, 99)
    if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
        chn = np.clip((chf - lo) / (hi - lo), 0.0, 1.0)
    else:
        chn = np.clip(chf / (chf.max() if chf.max() > 0 else 1.0), 0.0, 1.0)
    return (chn * 255).astype(np.uint8)

def draw_contours_overlay_rgb(rgb_img, mask, edge_color=(255,0,0)):
    """Draw binary-mask contours onto an RGB uint8 image (returns a copy)."""
    if rgb_img is None:
        return None
    rgb = rgb_img.copy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
    contours = measure.find_contours((mask > 0).astype(np.uint8), 0.5)
    for c in contours:
        pts = np.round(c).astype(int)
        pts[:,0] = np.clip(pts[:,0], 0, rgb.shape[0]-1)
        pts[:,1] = np.clip(pts[:,1], 0, rgb.shape[1]-1)
        # set color on the contour pixels
        rgb[pts[:,0], pts[:,1], 0] = edge_color[2]
        rgb[pts[:,0], pts[:,1], 1] = edge_color[1]
        rgb[pts[:,0], pts[:,1], 2] = edge_color[0]
    return rgb

def draw_contours_on_gray(gray, mask, val=255):
    overlay = cv2.cvtColor((gray).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours = measure.find_contours((mask > 0).astype(np.uint8), 0.5)
    for c in contours:
        pts = np.round(c).astype(int)
        pts[:,0] = np.clip(pts[:,0], 0, overlay.shape[0]-1)
        pts[:,1] = np.clip(pts[:,1], 0, overlay.shape[1]-1)
        overlay[pts[:,0], pts[:,1], :] = val
    return overlay

# crop helper consistent with earlier pipeline: crop_center(img, x, y, out_w)
def crop_center(img, x, y, out_w):
    edge = int(round((out_w - 1) / 2))
    x = int(round(x))
    y = int(round(y))
    pad_val = 0
    if img.ndim == 3:
        pad = [pad_val] * img.shape[2]
    else:
        pad = pad_val
    padded = cv2.copyMakeBorder(img, edge, edge, edge, edge, cv2.BORDER_CONSTANT, value=pad)
    x_p = x + edge
    y_p = y + edge
    return padded[(y_p - edge):(y_p + edge + 1), (x_p - edge):(x_p + edge + 1)]

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--slide_id', required=True)
    p.add_argument('--frame_id', required=True, type=int)
    p.add_argument('--cell_id', required=True)
    p.add_argument('--cell_x', required=True, type=float)
    p.add_argument('--cell_y', required=True, type=float)
    p.add_argument('--input-h5', default=None, help='HDF5 that may contain images, features, masks')
    p.add_argument('--cellpose-model', default=None, help='Path to Cellpose pretrained model (optional)')
    p.add_argument('--starts', default='1,2305,4609,9217', help='comma-separated starts for tile channels')
    p.add_argument('--outdir', default='inspect_out', help='output dir for PNGs')
    p.add_argument('--device', default='cuda:0', help='torch device string')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    starts = [int(x) for x in args.starts.split(',')]

    matched_index = None
    hdf5_mask = None
    features_row = None

    if args.input_h5:
        print("[info] Opening input HDF5:", args.input_h5)
        try:
            df = pd.read_hdf(args.input_h5, key='features')
            cond = (df['slide_id'].astype(str) == str(args.slide_id)) & (df['frame_id'].astype(int) == int(args.frame_id))
            if 'cell_id' in df.columns:
                cond = cond & (df['cell_id'].astype(str) == str(args.cell_id))
            candidates = df[cond]
            if len(candidates) > 0:
                matched_index = candidates.index[0]
                features_row = candidates.loc[matched_index]
                print(f"[info] matched features row index: {matched_index}")
            else:
                # fallback nearest by proximity if x,y exist
                if 'x' in df.columns and 'y' in df.columns:
                    dxy = np.sqrt((df['x'] - args.cell_x)**2 + (df['y'] - args.cell_y)**2)
                    nearest_idx = int(dxy.idxmin())
                    if dxy.loc[nearest_idx] < 5.0:
                        matched_index = nearest_idx
                        features_row = df.loc[matched_index]
                        print(f"[info] fallback matched features row index by proximity: {matched_index}")
        except Exception as e:
            print("[warn] couldn't read features:", e)

        # try read masks if present
        try:
            with h5py.File(args.input_h5, 'r') as f:
                if 'masks' in f and matched_index is not None:
                    mm = f['masks'][matched_index]
                    hdf5_mask = (mm > 0).astype(np.uint8)
                    print("[info] loaded HDF5 crop mask (shape):", hdf5_mask.shape)
        except Exception as e:
            print("[warn] couldn't read masks from HDF5:", e)

    # read full tile from disk
    tile = read_full_tile(args.slide_id, args.frame_id, starts)
    if tile is None:
        print(f"[error] Could not read tile for slide {args.slide_id} frame {args.frame_id}. Check tile paths and starts.")
    else:
        print("[info] read tile with shape, dtype:", tile.shape, tile.dtype)

    # run cellpose on tile (if model given and tile present)
    tile_mask = None
    cp_labels = None
    if tile is not None and args.cellpose_model:
        print("[info] loading Cellpose model...")
        device = args.device
        cp_model = models.CellposeModel(gpu=(str(device) != 'cpu'), pretrained_model=args.cellpose_model, device=torch.device(device))
        try:
            rgb_for_cp = channels_to_bgr(tile, [0,3], [2,3], [1,3])  # pipeline mapping
        except Exception:
            # fallback: use first 3 channels
            rgb_for_cp = tile[..., :3] if tile.ndim == 3 and tile.shape[2] >= 3 else tile
        print("[info] running Cellpose (this can take a while)...")
        cp_labels, _, _ = cp_model.eval(rgb_for_cp, diameter=20, channels=[0,0], batch_size=8)
        tile_mask = (cp_labels > 0).astype(np.uint8)
        print("[info] Cellpose labels unique:", np.unique(cp_labels)[:10])

    # Build per-channel display images
    comp_rgb = None
    ch_imgs = []
    if tile is not None:
        for ch in range(tile.shape[2]):
            ch_imgs.append(normalize_display_channel(tile[..., ch]))
        # build composite using channels_to_bgr mapping, then convert to RGB uint8
        try:
            comp_bgr = channels_to_bgr(tile, [0,3], [2,3], [1,3])
            # normalize each channel of comp_bgr to uint8 if not already
            if comp_bgr.dtype != np.uint8:
                comp_bgr_chs = []
                for k in range(comp_bgr.shape[2]):
                    comp_bgr_chs.append(normalize_display_channel(comp_bgr[..., k]))
                comp_bgr = cv2.merge(comp_bgr_chs)
            comp_rgb = comp_bgr[..., ::-1]  # BGR->RGB for matplotlib
        except Exception:
            comp_rgb = cv2.merge([ch_imgs[2], ch_imgs[1], ch_imgs[0]])  # fallback R,G,B

    # --- TILE ROW VISUAL (long horizontal) ---
    tile_out = os.path.join(args.outdir, f"{args.slide_id}_f{args.frame_id}_c{args.cell_id}_tile_row.png")
    # We'll make a single-row figure with: [full composite (large)] [tile outlines small] [ch0] [ch1] [ch2] [ch3]
    n_panels = 6
    fig_h = 4
    fig_w = 18
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, n_panels, width_ratios=[3,1,1,1,1,1])

    # big composite
    ax0 = fig.add_subplot(gs[0,0])
    if comp_rgb is not None:
        ax0.imshow(comp_rgb)
        ax0.set_title(f"Full tile composite (RGB) slide:{args.slide_id} frame:{args.frame_id}")
        ax0.scatter([args.cell_x], [args.cell_y], s=40, facecolors='none', edgecolors='cyan', linewidth=1.5)
        if tile_mask is not None:
            overlay = draw_contours_overlay_rgb(comp_rgb, tile_mask, edge_color=(255,0,0))
            ax0.imshow(overlay, alpha=0.6)
    else:
        ax0.text(0.5,0.5,"no tile",ha='center',va='center')
    ax0.axis('off')

    # tile outlines (small)
    ax1 = fig.add_subplot(gs[0,1])
    if comp_rgb is not None:
        ax1.imshow(comp_rgb)
        if tile_mask is not None:
            overlay = draw_contours_overlay_rgb(comp_rgb, tile_mask, edge_color=(255,100,100))
            ax1.imshow(overlay, alpha=0.8)
    ax1.set_title("Tile outlines")
    ax1.axis('off')

    # channel panels
    for i in range(4):
        ax = fig.add_subplot(gs[0, 2 + i])
        if tile is not None:
            gray = ch_imgs[i]
            if tile_mask is not None:
                overlay_gray = draw_contours_on_gray(gray, tile_mask, val=200)
            else:
                overlay_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            ax.imshow(overlay_gray)
            ax.set_title(f"ch{i} (raw)")
        else:
            ax.text(0.5,0.5,"no tile",ha='center',va='center')
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(tile_out, dpi=180)
    plt.close(fig)
    print("[info] saved tile row visualization ->", tile_out)

    # --- CROP 225x225 VISUAL (SEPARATE FILE) ---
    crop_out = os.path.join(args.outdir, f"{args.slide_id}_f{args.frame_id}_c{args.cell_id}_crop225.png")

    # produce 225x225 RGB crop (normalized)
    crop225_rgb = None
    crop225_ch = []
    if comp_rgb is not None:
        crop225_rgb = crop_center(comp_rgb, args.cell_x, args.cell_y, 225)
    elif tile is not None:
        # construct RGB if composite missing
        crop_rgb_alt = cv2.merge([ch_imgs[2], ch_imgs[1], ch_imgs[0]])
        crop225_rgb = crop_center(crop_rgb_alt, args.cell_x, args.cell_y, 225)

    # tile-derived mask cropped to 225x225 (so overlays align)
    tile_mask_crop225 = None
    if tile_mask is not None:
        tile_mask_crop225 = crop_center(tile_mask, args.cell_x, args.cell_y, 225)

    # tile-derived mask cropped to 75x75 (for separate display if desired)
    tile_mask_crop75 = None
    if tile_mask is not None:
        tile_mask_crop75 = crop_center(tile_mask, args.cell_x, args.cell_y, 75)

    # HDF5 crop mask (if present) scaled to 75x75 or padded
    hdf5_mask_display = None
    if hdf5_mask is not None:
        # ensure it's 75x75; if it is smaller/larger, center-crop or pad
        mm = np.asarray(hdf5_mask)
        if mm.shape == (75,75):
            hdf5_mask_display = mm
        else:
            # if mm is single-channel crop of different size, try center-crop/pad to 75
            hdf5_mask_display = np.zeros((75,75), dtype=np.uint8)
            hh = min(mm.shape[0], 75)
            ww = min(mm.shape[1], 75)
            # place centered
            y0 = (75 - hh)//2
            x0 = (75 - ww)//2
            hdf5_mask_display[y0:y0+hh, x0:x0+ww] = (mm[:hh, :ww] > 0).astype(np.uint8)

    # Build crop figure
    fig2 = plt.figure(figsize=(8,6))
    gs2 = fig2.add_gridspec(2, 3, width_ratios=[1,1,1], height_ratios=[1,1])

    # Panel 1: 225x225 composite with overlayed 225 mask outlines and 75x75 yellow box
    axc = fig2.add_subplot(gs2[0, 0])
    if crop225_rgb is not None:
        axc.imshow(crop225_rgb)
        axc.set_title("225x225 crop (yellow = 75x75)")
        # draw 75x75 rectangle centered
        edge = int(round((75 - 1) / 2))
        cx = crop225_rgb.shape[1] // 2
        cy = crop225_rgb.shape[0] // 2
        top_left_x = cx - edge
        top_left_y = cy - edge
        rect = Rectangle((top_left_x, top_left_y), 75, 75, linewidth=2, edgecolor='yellow', facecolor='none')
        axc.add_patch(rect)
        # center marker
        axc.scatter([cx], [cy], s=40, c='cyan', marker='x')
        # overlay tile_mask_crop225 if available
        if tile_mask_crop225 is not None:
            overlay = draw_contours_overlay_rgb(crop225_rgb, tile_mask_crop225, edge_color=(255,0,0))
            axc.imshow(overlay, alpha=0.6)
    else:
        axc.text(0.5,0.5,"no crop available",ha='center',va='center')
    axc.axis('off')

    # Panel 2: zoom into the 75x75 region from the tile-derived mask (if available) + composite zoom
    axc2 = fig2.add_subplot(gs2[0,1])
    if crop225_rgb is not None:
        crop75_rgb = crop_center(crop225_rgb, crop225_rgb.shape[1]//2, crop225_rgb.shape[0]//2, 75)
        axc2.imshow(crop75_rgb)
        if tile_mask_crop75 is not None:
            overlay75 = draw_contours_overlay_rgb(crop75_rgb, tile_mask_crop75, edge_color=(255,0,0))
            axc2.imshow(overlay75, alpha=0.6)
        axc2.set_title("75x75 zoom (tile-derived)")
    else:
        axc2.text(0.5,0.5,"no crop75",ha='center',va='center')
    axc2.axis('off')

    # Panel 3: HDF5 crop mask (if present)
    axh = fig2.add_subplot(gs2[0,2])
    if hdf5_mask_display is not None:
        axh.imshow(hdf5_mask_display, cmap='gray')
        axh.set_title("HDF5 crop mask (if available)")
    else:
        axh.text(0.5,0.5,"no HDF5 crop mask",ha='center',va='center')
    axh.axis('off')

    # Optionally show tile_mask_crop225 as standalone grayscale (big)
    axt = fig2.add_subplot(gs2[1,0])
    if tile_mask_crop225 is not None:
        axt.imshow(tile_mask_crop225, cmap='gray')
        axt.set_title("Tile-derived mask (225x225)")
    else:
        axt.text(0.5,0.5,"no tile mask",ha='center',va='center')
    axt.axis('off')

    # show three channel grayscale crops for visual verification (use center)
    for i in range(3):
        ax = fig2.add_subplot(gs2[1, 1 + i - 1]) if i < 2 else fig2.add_subplot(gs2[1, 2])
        if tile is not None:
            # choose channel indices safe
            ch_idx = i if i < tile.shape[2] else 0
            ch_full = normalize_display_channel(tile[..., ch_idx])
            ch_crop = crop_center(ch_full, args.cell_x, args.cell_y, 75)
            if ch_crop is None:
                ax.text(0.5,0.5,"no channel",ha='center',va='center')
            else:
                ax.imshow(ch_crop, cmap='gray')
            ax.set_title(f"ch{ch_idx} (75x75)")
        else:
            ax.text(0.5,0.5,"no tile",ha='center',va='center')
        ax.axis('off')

    plt.tight_layout()
    fig2.savefig(crop_out, dpi=160)
    plt.close(fig2)
    print("[info] saved crop visualization ->", crop_out)

    # Also save a simple tile overlay (BGR) for quick viewing (cv2)
    if tile_mask is not None and comp_rgb is not None:
        overlay_bgr = draw_contours_overlay_rgb(comp_rgb, tile_mask, edge_color=(255,0,0))[..., ::-1]  # RGB->BGR
        quick_overlay_path = os.path.join(args.outdir, f"{args.slide_id}_f{args.frame_id}_c{args.cell_id}_tile_overlay.png")
        cv2.imwrite(quick_overlay_path, overlay_bgr)
        print("[info] saved quick tile overlay (cv2) ->", quick_overlay_path)

    print("[done]")

if __name__ == '__main__':
    main()