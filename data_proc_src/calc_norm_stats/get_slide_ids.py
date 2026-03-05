#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
import sys

def extract_slide_ids(h5_dir, output_file):
    h5_dir_path = Path(h5_dir)
    if not h5_dir_path.exists() or not h5_dir_path.is_dir():
        print(f"Error: Directory '{h5_dir}' does not exist.")
        sys.exit(1)

    # Find all .hdf5 files recursively
    print(f"Scanning for .hdf5 files in {h5_dir_path}...")
    h5_files = list(h5_dir_path.rglob("*.hdf5"))
    print(f"Found {len(h5_files)} HDF5 files.")

    all_slide_ids = set()

    for i, h5_path in enumerate(h5_files, 1):
        sys.stdout.write(f"\rProcessing file {i}/{len(h5_files)}: {h5_path.name}...")
        sys.stdout.flush()
        
        try:
            # Efficiently read only the 'slide_id' column from the 'features' table
            df = pd.read_hdf(h5_path, key='features', columns=['slide_id'])
            
            # Extract unique IDs and add them to our master set
            unique_ids = df['slide_id'].dropna().astype(str).unique()
            
            # Clean up the IDs (e.g., remove any _01 suffixes if they exist in your raw data)
            cleaned_ids = {sid.split("_")[0] for sid in unique_ids}
            all_slide_ids.update(cleaned_ids)
            
        except KeyError:
            # The 'features' key or 'slide_id' column doesn't exist in this file
            print(f"\n  -> Warning: 'features' table or 'slide_id' column missing in {h5_path.name}")
        except Exception as e:
            print(f"\n  -> Error reading {h5_path.name}: {e}")

    print("\n\nExtraction complete!")
    print(f"Found {len(all_slide_ids)} unique slide IDs.")

    # Save the results to a text file
    with open(output_file, "w") as f:
        for slide_id in sorted(all_slide_ids):
            f.write(f"{slide_id}\n")
            
    print(f"Successfully saved slide IDs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract unique slide_ids from a directory of HDF5 files.")
    parser.add_argument("--h5_dir", required=True, help="Root directory containing your .hdf5 files")
    parser.add_argument("--output", default="slides.txt", help="Output text file path (default: slides.txt)")
    
    args = parser.parse_args()
    extract_slide_ids(args.h5_dir, args.output)

if __name__ == "__main__":
    main()