#!/usr/bin/env python3
import argparse
import subprocess
import pandas as pd
import time
import sys
import json
from pathlib import Path
from collections import OrderedDict
import multiprocessing
import h5py
import os

# ==========================
# Config
# ==========================
DEFAULT_REFRESH = 30
DEFAULT_SHARD_ROWS = 100_000

# ==========================
# Utilities
# ==========================
def sanitize_type_name(s):
    return s.strip().replace(" ", "_").replace("/", "_").replace("-", "_")

def clear_screen():
    sys.stdout.write("\033[2J\033[H")

def format_eta(seconds):
    if seconds is None or seconds <= 0:
        return "-"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"

def read_progress_json(path: Path):
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            return int(json.load(f).get("images_written", 0))
    except Exception:
        return 0

def safe_count_images(h5_path: Path):
    if not h5_path.exists():
        return 0
    try:
        with h5py.File(h5_path, "r") as f:
            if "images" in f:
                return int(f["images"].shape[0])
    except Exception:
        return None
    return 0

def read_skipped_sets(out_h5: Path):
    skipped_slides = set()
    skipped_frames = set()

    slides_txt = out_h5.with_name(out_h5.stem + "_skipped_slides.txt")
    frames_txt = out_h5.with_name(out_h5.stem + "_skipped_frames.txt")

    if slides_txt.exists():
        skipped_slides = set(slides_txt.read_text().splitlines())

    if frames_txt.exists():
        for line in frames_txt.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                skipped_frames.add((parts[0], int(parts[1])))

    return skipped_slides, skipped_frames

# ==========================
# Sharding (slide-aware)
# ==========================
def shard_by_slide_id_target_rows(df, target_rows):
    groups = list(df.groupby("slide_id"))
    groups.sort(key=lambda x: len(x[1]), reverse=True)

    shards = []
    current = []
    count = 0

    for _, g in groups:
        if count + len(g) > target_rows and current:
            shards.append(pd.concat(current))
            current, count = [], 0
        current.append(g)
        count += len(g)

    if current:
        shards.append(pd.concat(current))

    return shards

# ==========================
# Dashboard
# ==========================
def print_dashboard(status):
    clear_screen()
    print(f"{'Job':45} {'State':10} {'Processed':20} {'ETA'}")
    print("-" * 100)
    for job, s in status.items():
        print(
            f"{job:45} {s['state']:10} "
            f"{s['done']:,} / {s['total']:,} {s['eta']}"
        )
    print("-" * 100)
    sys.stdout.flush()

# ==========================
# Main
# ==========================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--extract-script", required=True)
    p.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    p.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    p.add_argument("--refresh", type=int, default=DEFAULT_REFRESH)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()

    df = pd.read_csv(args.csv, dtype={"slide_id": str})
    df["slide_id"] = df["slide_id"].astype(str).str.split("_", n=1).str[0]

    out_root = Path(args.out_root)
    tmp_dir = out_root / "_tmp_csvs"
    tmp_dir.mkdir(exist_ok=True)

    status = OrderedDict()
    queue = []
    running = {}

    # ===============================
    # Build jobs + skip completed
    # ===============================
    for type_name, g in df.groupby("type"):
        base = sanitize_type_name(type_name)
        out_dir = out_root / base
        out_dir.mkdir(exist_ok=True)

        full_h5 = out_dir / f"{base}.hdf5"
        full_images = safe_count_images(full_h5)
        if full_images is None:
            full_images = read_progress_json(full_h5.with_suffix(".hdf5.progress.json"))

        skipped_slides, skipped_frames = read_skipped_sets(full_h5)

        # remove skipped slides
        g_eff = g[~g["slide_id"].isin(skipped_slides)]

        if len(g_eff) <= args.shard_rows:
            shards = [(base, g_eff)]
        else:
            shards = [
                (f"{base}:shard_{i:02d}", shard_df)
                for i, shard_df in enumerate(
                    shard_by_slide_id_target_rows(g_eff, args.shard_rows)
                )
            ]

        for job, shard_df in shards:
            shard_h5 = out_dir / f"{job}.hdf5"
            shard_images = safe_count_images(shard_h5)
            if shard_images is None:
                shard_images = read_progress_json(shard_h5.with_suffix(".hdf5.progress.json"))

            processed = max(full_images or 0, shard_images or 0)
            total = len(shard_df)

            state = "DONE" if processed >= total else "QUEUED"

            status[job] = {
                "state": state,
                "done": processed,
                "total": total,
                "start": None,
                "eta": "-"
            }

            if state != "DONE":
                queue.append((job, shard_df, base))

    # ===============================
    # Execution loop
    # ===============================
    while queue or running:
        while queue and len(running) < args.jobs:
            job, shard_df, base = queue.pop(0)

            tmp_csv = tmp_dir / f"{job}.csv"
            shard_df.to_csv(tmp_csv, index=False)

            out_h5 = out_root / base / f"{job}.hdf5"

            cmd = [
                sys.executable,
                args.extract_script,
                "--data", str(tmp_csv),
                "--output", str(out_h5)
            ]
            if args.verbose:
                cmd.append("-" + "v" * args.verbose)

            p = subprocess.Popen(cmd)
            running[job] = {
                "proc": p,
                "start": time.time(),
                "h5": out_h5,
                "progress": out_h5.with_suffix(".hdf5.progress.json")
            }

            status[job]["state"] = "RUNNING"
            status[job]["start"] = time.time()

        for job, info in list(running.items()):
            img = safe_count_images(info["h5"])
            if img is None:
                img = read_progress_json(info["progress"])

            status[job]["done"] = img

            if status[job]["start"] and img > 0:
                rate = img / (time.time() - status[job]["start"])
                remaining = status[job]["total"] - img
                status[job]["eta"] = format_eta(remaining / rate)

            if info["proc"].poll() is not None:
                status[job]["state"] = "DONE"
                status[job]["eta"] = "-"
                running.pop(job)

        print_dashboard(status)
        time.sleep(args.refresh)

if __name__ == "__main__":
    main()
