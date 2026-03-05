#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import json
import time
from pathlib import Path

def aggregate_stats(stats_dir, output_file):
    """Combines all individual slide JSONs into one master dictionary."""
    print(f"\n--- Aggregating currently finished stats into {output_file} ---")
    master_dict = {}
    for json_path in Path(stats_dir).glob("stats_*.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                slide_id = data.get("slide_id")
                if slide_id:
                    master_dict[slide_id] = data["channels"]
        except Exception as e:
            print(f"Failed to read {json_path}: {e}")
            
    with open(output_file, "w") as f:
        json.dump(master_dict, f, indent=4)
    print(f"--- Aggregation complete. Total slides in master: {len(master_dict)} ---\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slides_txt", required=True, help="Path to text file with slide IDs (one per line)")
    p.add_argument("--out_dir", required=True, help="Directory to store individual JSONs")
    p.add_argument("--master_out", required=True, help="Path for the combined master JSON file")
    p.add_argument("--jobs", type=int, default=8, help="Number of parallel processes")
    p.add_argument("--n_tiles", type=int, default=100, help="Number of tiles to sample per slide")
    p.add_argument("--k_aggregate", type=int, default=10, help="Aggregate master file every K completed slides")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read slides
    with open(args.slides_txt, "r") as f:
        slides = [line.strip() for line in f if line.strip()]
        
    # Clean slide IDs (in case they have _01 suffixes in the txt file)
    slides = [s.split("_")[0] for s in slides]

    queue = slides.copy()
    running_procs = {}
    completed_count = 0
    last_aggregated_count = 0

    print(f"Found {len(queue)} slides. Starting {args.jobs} parallel workers.")

    while queue or running_procs:
        # Fill the process queue up to the parallel limit
        while queue and len(running_procs) < args.jobs:
            slide_id = queue.pop(0)
            
            cmd = [
                sys.executable, "calc_slide_stats.py",
                "--slide_id", slide_id,
                "--output_dir", args.out_dir,
                "--n_tiles", str(args.n_tiles)
            ]
            
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running_procs[slide_id] = proc

        # Check for finished processes
        finished_slides = []
        for slide_id, proc in list(running_procs.items()):
            if proc.poll() is not None:  # Process finished
                finished_slides.append(slide_id)
                del running_procs[slide_id]
                completed_count += 1
                sys.stdout.write(f"\rProgress: {completed_count}/{len(slides)} completed. Running: {len(running_procs)}   ")
                sys.stdout.flush()

        # Check if we hit the 'k' threshold to aggregate
        if (completed_count - last_aggregated_count) >= args.k_aggregate:
            aggregate_stats(args.out_dir, args.master_out)
            last_aggregated_count = completed_count

        time.sleep(1.0) # Prevent CPU spinning

    # Final aggregation when everything is completely done
    aggregate_stats(args.out_dir, args.master_out)
    print("All slides processed successfully.")

if __name__ == "__main__":
    main()