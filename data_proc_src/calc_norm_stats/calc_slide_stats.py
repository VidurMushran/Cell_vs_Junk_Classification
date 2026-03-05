#!/usr/bin/env python3
import argparse
import os
import glob
import json
import numpy as np
import random
from pathlib import Path
try:
    import tifffile
except ImportError:
    print("Please install tifffile: pip install tifffile")
    exit(1)

def find_slide_directory(slide_id):
    """Reused from your extraction script to locate the slide images."""
    tube_id = slide_id[:5]
    for d in os.listdir("/mnt"):
        base = os.path.join("/mnt", d)
        pattern = os.path.join(
            base, "Oncoscope", f"tubeID_{tube_id}", "*",
            f"slideID_{slide_id}", "bzScanner", "proc"
        )
        matches = glob.glob(pattern)
        if matches:
            image_dir = matches[0]
            if glob.glob(os.path.join(image_dir, "Tile*.tif")):
                return image_dir, "Tile%06d.tif"
            if glob.glob(os.path.join(image_dir, "Tile*.jpg")):
                return image_dir, "Tile%06d.jpg"
    return None, None

def compute_channel_stats(pixels):
    """Calculates all metrics needed for the various normalization methods."""
    if len(pixels) == 0:
        return {}

    # Percentiles for Robust Scaling
    p1, p5, p50, p95, p99, p99_9 = np.percentile(pixels, [1, 5, 50, 95, 99, 99.9])
    
    # RESTORE / Background approximation
    # Create a histogram to find the most common intensity (the background mode)
    hist, bins = np.histogram(pixels, bins=1000, range=(0, 65535))
    mode_idx = np.argmax(hist)
    bg_mode = (bins[mode_idx] + bins[mode_idx+1]) / 2.0
    
    # Calculate spread of the background (left side of the mode)
    left_of_mode = pixels[pixels <= bg_mode]
    bg_std_approx = np.std(left_of_mode) if len(left_of_mode) > 0 else 0.0

    return {
        "global_min": float(np.min(pixels)),
        "global_max": float(np.max(pixels)),
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "p01_0": float(p1),
        "p05_0": float(p5),
        "p50_0": float(p50),
        "p95_0": float(p95),
        "p99_0": float(p99),
        "p99_9": float(p99_9),
        "bg_mode": float(bg_mode),
        "bg_std_approx": float(bg_std_approx)
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slide_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_tiles", type=int, default=100, help="Number of random tiles to sample")
    p.add_argument("--channels", nargs="+", default=["DAPI", "TRITC", "CY5", "FITC"])
    p.add_argument("--starts", nargs="+", type=int, default=[1, 2305, 4609, 9217])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"stats_{args.slide_id}.json")
    
    if os.path.exists(out_file):
        print(f"[{args.slide_id}] Stats already exist. Skipping.")
        return

    image_dir, fmt = find_slide_directory(args.slide_id)
    if not image_dir:
        print(f"[{args.slide_id}] ERROR: Slide directory not found.")
        return

    # Determine max available base tile (frame_id) by looking at the distance between starts
    # Assuming start[0] is 1, and start[1] is 2305, max base tiles is 2304.
    max_base_tile = args.starts[1] - args.starts[0] if len(args.starts) > 1 else 2304
    
    # Randomly sample n frame_ids
    sampled_frames = random.sample(range(1, max_base_tile + 1), min(args.n_tiles, max_base_tile))
    
    slide_stats = {"slide_id": args.slide_id, "channels": {}}

    for ch_idx, (ch_name, start_offset) in enumerate(zip(args.channels, args.starts)):
        print(f"[{args.slide_id}] Processing {ch_name}...")
        channel_pixels = []
        
        for frame_id in sampled_frames:
            tile_path = os.path.join(image_dir, fmt % (frame_id + start_offset - 1))
            if os.path.exists(tile_path):
                try:
                    # Load image, flatten it, and downsample it slightly to save RAM during concat
                    img = tifffile.imread(tile_path)
                    # Taking every 4th pixel is mathematically sufficient for statistical distribution
                    channel_pixels.append(img[::4, ::4].flatten()) 
                except Exception as e:
                    pass

        if channel_pixels:
            all_pixels = np.concatenate(channel_pixels)
            slide_stats["channels"][ch_name] = compute_channel_stats(all_pixels)
        else:
            slide_stats["channels"][ch_name] = None

    with open(out_file, "w") as f:
        json.dump(slide_stats, f, indent=4)
    print(f"[{args.slide_id}] Finished writing {out_file}")

if __name__ == "__main__":
    main()