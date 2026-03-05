#!/usr/bin/env python3
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes cellx/celly to cell_x/cell_y and removes 'Unnamed' index columns."""
    df.columns = [str(c).strip() for c in df.columns]
    
    df = df.drop(
        columns=[c for c in df.columns if c.lower().startswith("unnamed")],
        errors="ignore"
    )
    
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["x", "cellx"]:
            rename_map[c] = "cell_x"
        elif cl in ["y", "celly"]:
            rename_map[c] = "cell_y"
    return df.rename(columns=rename_map)

def load_manifest_files(target_dir: Path):
    manifests = list(target_dir.glob("*.manifest.parquet"))
    dfs = []
    for p in manifests:
        try:
            df = pd.read_parquet(p)
            df = normalize_columns(df)
            df["_source_h5_guess"] = p.with_suffix("").with_suffix(".hdf5").name
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Could not read {p}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def generate_manifest_from_features(h5_path: Path) -> pd.DataFrame:
    """If an HDF5 file lacks a manifest, read its features table to infer one."""
    print(f"[INFO] No manifest found for {h5_path.name}. Inferring manifest directly from its /features table...")
    try:
        df = pd.read_hdf(h5_path, key="features")
    except Exception as e:
        print(f"[ERROR] Could not read /features from {h5_path.name} to generate manifest. Error: {e}")
        return pd.DataFrame()
        
    df = normalize_columns(df)
    
    # The physical index in the HDF5 file is strictly its row number (0 to len-1)
    df["hdf5_index"] = np.arange(len(df))
        
    df["status"] = "included"
    df["reason"] = "inferred_from_existing_features"
    df["_source_h5_guess"] = h5_path.name
    
    return df

def auto_repair_shard(h5_path: Path, manifest_df: pd.DataFrame):
    """Automatically rebuilds the pandas /features table if it is missing or corrupted."""
    print(f"[REPAIR] Rebuilding missing/corrupt features table for {h5_path.name}...")
    shard_manifest = manifest_df[(manifest_df["_source_h5_guess"] == h5_path.name) & (manifest_df["status"] == "included")].copy()
    
    if shard_manifest.empty:
        print(f"[REPAIR-SKIP] No included rows in manifest for {h5_path.name}.")
        return

    shard_manifest = shard_manifest.sort_values("hdf5_index").reset_index(drop=True)
    shard_manifest["image_id"] = shard_manifest["hdf5_index"]
    shard_manifest["__source_h5__"] = h5_path.name

    with pd.HDFStore(h5_path, mode="a") as store:
        if "features" in store:
            store.remove("features")
        store.put("features", shard_manifest, format="table", data_columns=True, index=False)
    print(f"[REPAIR] Successfully rebuilt features for {h5_path.name}.")

def clean_hdf5_features(h5_path: Path):
    """Removes verbose PyTables _i_table indices from a given HDF5 file."""
    print(f"[CLEAN] Stripping PyTables indices from {h5_path.name}...")
    try:
        df = pd.read_hdf(h5_path, key="features")
        with pd.HDFStore(h5_path, mode="a") as store:
            if "features" in store:
                store.remove("features")
            store.put("features", df, format="table", data_columns=True, index=False)
        print(f"[CLEAN] Successfully cleaned {h5_path.name}.")
    except Exception as e:
        print(f"[WARNING] Could not clean {h5_path.name}. It may need repair first. Error: {e}")

def normalize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    int_columns = ["hdf5_index", "final_hdf5_index", "image_id"]
    for col in int_columns:
        if col in df.columns:
            ser = df[col]
            missing_mask = ser.isna()
            if missing_mask.any():
                df[f"{col}_is_missing"] = missing_mask.astype("uint8")
                df[col] = ser.fillna(-1).astype("int64")
            else:
                df[col] = ser.astype("int64")
    return df

def generate_report(type_dir: Path, manifest: pd.DataFrame, baseline_csv: str):
    """Generates a one-file report comparing the baseline to the processed manifest."""
    print("Generating baseline reconciliation report...")
    baseline_df = pd.read_csv(baseline_csv)
    baseline_df = normalize_columns(baseline_df)
    
    if "type" in baseline_df.columns:
        baseline_df = baseline_df[baseline_df["type"] == type_dir.name]

    merge_keys = ["slide_id", "frame_id", "cell_id", "cell_x", "cell_y"]
    missing_keys = [k for k in merge_keys if k not in baseline_df.columns or k not in manifest.columns]
    if missing_keys:
        print(f"[WARNING] Missing keys for report generation: {missing_keys}. Falling back to standard keys.")
        merge_keys = ["slide_id", "frame_id", "cell_id"]

    report = baseline_df.merge(
        manifest[merge_keys + ["status", "reason"]], 
        on=merge_keys, 
        how="left"
    )
    report["status"] = report["status"].fillna("missing_from_manifest")
    report["reason"] = report["reason"].fillna("pipeline_failed_to_process")
    
    report_path = type_dir / f"{type_dir.name}_merge_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Report saved to: {report_path}")

def merge_type(type_dir: Path, out_name=None, reset=False, baseline_csv=None, clean_shards=False):
    type_dir = Path(type_dir)
    out_name = out_name or f"{type_dir.name}.final.hdf5"
    out_path = type_dir / out_name

    shards_dir = type_dir / "shards"
    search_dir = shards_dir if shards_dir.exists() else type_dir

    if out_path.exists() and not reset:
        print(f"{out_path} exists. Use --reset.")
        return

    if out_path.exists():
        out_path.unlink()

    source_h5s = [p for p in search_dir.glob("*.hdf5") if not p.name.endswith(".final.hdf5")]
    h5_map = {p.name: p for p in source_h5s}

    if clean_shards:
        print("\n--- Cleaning Shards ---")
        for p in source_h5s:
            clean_hdf5_features(p)
        print("--- Finished Cleaning Shards ---\n")

    manifest = load_manifest_files(search_dir)
    if manifest is None:
        manifest = pd.DataFrame()

    existing_sources = set(manifest["_source_h5_guess"].unique()) if not manifest.empty else set()
    extra_manifests = []
    
    for p in source_h5s:
        if p.name not in existing_sources:
            inferred_manifest = generate_manifest_from_features(p)
            if not inferred_manifest.empty:
                extra_manifests.append(inferred_manifest)
                
    if extra_manifests:
        manifest = pd.concat([manifest] + extra_manifests, ignore_index=True)

    if manifest.empty:
        print("No manifests or features tables found. Nothing to merge.")
        return

    dedup_keys = ["slide_id", "frame_id", "cell_id"]
    if "cell_x" in manifest.columns and "cell_y" in manifest.columns:
        dedup_keys.extend(["cell_x", "cell_y"])
        
    manifest = manifest.drop_duplicates(subset=dedup_keys, keep="first")

    if baseline_csv:
        generate_report(type_dir, manifest, baseline_csv)

    included = manifest[manifest.status == "included"].copy()

    if included.empty:
        print("No included rows to merge.")
        return

    img_shape, img_dtype, channels_arr = None, None, None

    for p in source_h5s:
        try:
            with h5py.File(p, "r") as f:
                if "images" in f:
                    img_shape = f["images"].shape[1:]
                    img_dtype = f["images"].dtype
                    if "channels" in f:
                        channels_arr = f["channels"][:]
                    break
        except Exception:
            continue

    if img_shape is None:
        print("Could not determine image shape.")
        return

    with h5py.File(out_path, "w") as out_f:
        out_f.create_dataset(
            "images", shape=(0,) + img_shape, maxshape=(None,) + img_shape,
            chunks=(256,) + img_shape, compression="gzip", compression_opts=4,
            shuffle=True, dtype=img_dtype
        )
        if channels_arr is not None:
            out_f.create_dataset("channels", data=channels_arr)

    master_columns = None
    master_dtypes = None
    final_index = 0

    for src_name, group_df in included.groupby("_source_h5_guess"):
        src_path = h5_map.get(src_name)
        if src_path is None:
            print(f"[WARNING] Missing source HDF5: {src_name}")
            continue

        print(f"Streaming from {src_name} ({len(group_df)} rows)")
        indices = group_df["hdf5_index"].dropna().astype(int).values
        if len(indices) == 0:
            continue

        try:
            pd.read_hdf(src_path, key="features", stop=1)
        except Exception:
            auto_repair_shard(src_path, manifest)

        order = np.argsort(indices)
        sorted_idx = indices[order]

        with h5py.File(src_path, "r") as src_f:
            src_images = src_f["images"]
            starts, ends = [sorted_idx[0]], []
            for a, b in zip(sorted_idx[:-1], sorted_idx[1:]):
                if b != a + 1:
                    ends.append(a + 1)
                    starts.append(b)
            ends.append(sorted_idx[-1] + 1)

            blocks = [src_images[s:e] for s, e in zip(starts, ends)]
            images_block = np.concatenate(blocks, axis=0)

        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        images_block = images_block[inv]

        with h5py.File(out_path, "a") as out_f:
            out_images = out_f["images"]
            n0 = out_images.shape[0]
            out_images.resize(n0 + len(images_block), axis=0)
            out_images[n0:n0 + len(images_block)] = images_block

        src_feat_df = pd.read_hdf(src_path, key="features")
        feat_sorted = src_feat_df.iloc[sorted_idx].reset_index(drop=True)
        feat_selected = feat_sorted.iloc[inv].reset_index(drop=True)

        feat_selected = normalize_columns(feat_selected)
        
        feat_selected["__source_h5__"] = src_name
        feat_selected["final_hdf5_index"] = np.arange(n0, n0 + len(feat_selected))
        feat_selected = normalize_feature_dtypes(feat_selected)

        # --- SCHEMA ALIGNMENT BLOCK ---
        # Lock in the schema from the first valid chunk
        if master_columns is None:
            master_columns = list(feat_selected.columns)
            master_dtypes = feat_selected.dtypes.to_dict()
        else:
            # Add missing columns safely based on original dtype
            for col in master_columns:
                if col not in feat_selected.columns:
                    dtype = master_dtypes[col]
                    if pd.api.types.is_integer_dtype(dtype):
                        feat_selected[col] = -1
                    elif pd.api.types.is_numeric_dtype(dtype):
                        feat_selected[col] = np.nan
                    else:
                        feat_selected[col] = ""
            
            # Enforce exact column order and drop extra columns
            feat_selected = feat_selected[master_columns]
            
            # Enforce exact types to prevent PyTables crashes
            for col in master_columns:
                if feat_selected[col].dtype != master_dtypes[col]:
                    try:
                        feat_selected[col] = feat_selected[col].astype(master_dtypes[col])
                    except Exception:
                        pass
        # ------------------------------

        feat_selected["__source_h5__"] = feat_selected["__source_h5__"].astype(str)
        min_itemsizes = {"__source_h5__": 255}
        
        if "slide_id" in feat_selected.columns:
            feat_selected["slide_id"] = feat_selected["slide_id"].astype(str)
            min_itemsizes["slide_id"] = 128

        data_cols = list(feat_selected.columns)

        with pd.HDFStore(out_path, mode="a") as store:
            store.append(
                "features", 
                feat_selected, 
                format="table", 
                data_columns=data_cols, 
                index=False,
                min_itemsize=min_itemsizes
            )

        final_index += len(indices)

    print(f"Finished streaming merge. Total rows: {final_index}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="type directory to merge")
    p.add_argument("--out-name")
    p.add_argument("--reset", action="store_true", help="Delete existing final hdf5 and rerun")
    p.add_argument("--baseline-csv", help="Original baseline CSV to generate comparison report")
    p.add_argument("--clean-shards", action="store_true", help="Remove PyTables indices from shards before merging")
    args = p.parse_args()

    merge_type(
        Path(args.dir), 
        out_name=args.out_name, 
        reset=args.reset, 
        baseline_csv=args.baseline_csv,
        clean_shards=args.clean_shards
    )

if __name__ == "__main__":
    main()