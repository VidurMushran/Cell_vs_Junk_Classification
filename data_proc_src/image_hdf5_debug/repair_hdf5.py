import h5py
import pandas as pd
import numpy as np
import os
import sys

# Disable locking to prevent "Resource temporarily unavailable" on network drives
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def repair_file(src_path):
    print(f"--- Repairing {src_path} ---")
    dst_path = src_path.replace(".hdf5", "_repaired.hdf5")
    
    # Clean up previous attempts
    if os.path.exists(dst_path):
        os.remove(dst_path)
    
    recovered_df = None
    
    # STEP 1: READ EVERYTHING (Don't write yet)
    print("Step 1: Reading source data...")
    try:
        with h5py.File(src_path, 'r') as f_src:
            # 1a. Try to recover features DataFrame
            if 'features' in f_src:
                try:
                    # Try Pandas read first
                    recovered_df = pd.read_hdf(src_path, key='features')
                except Exception as e:
                    print(f"Pandas read failed ({e}). Attempting raw recovery...")
                    # Raw recovery
                    feat_grp = f_src['features']
                    data_dict = {}
                    
                    # Check for 'fixed' format (column datasets)
                    for col_name in feat_grp.keys():
                        if col_name.startswith("_"): continue # Skip metadata
                        ds = feat_grp[col_name]
                        if isinstance(ds, h5py.Dataset):
                            # Skip 'table' if it exists, that's compound format handled below
                            if col_name == 'table': continue 
                            data_dict[col_name] = ds[:]
                    
                    # Check for 'table' format (Compound dataset)
                    if 'table' in feat_grp and isinstance(feat_grp['table'], h5py.Dataset):
                        print("Found compound 'table' dataset. Reading struct array...")
                        arr = feat_grp['table'][:]
                        # Convert structured array to dict
                        for name in arr.dtype.names:
                            if name == 'index': continue # We'll reset index anyway
                            data_dict[name] = arr[name]

                    if data_dict:
                        recovered_df = pd.DataFrame(data_dict)
                        # Handle byte strings
                        for c in recovered_df.select_dtypes([object]).columns:
                            try:
                                recovered_df[c] = recovered_df[c].str.decode('utf-8')
                            except: pass
                        print(f"Recovered {len(recovered_df)} rows via raw extraction.")
                    else:
                        print("Could not find raw columns in /features.")
            
            # 1b. Determine image count for validation
            n_images = f_src['images'].shape[0]
            
            # STEP 2: COPY RAW DATASETS TO NEW FILE
            print("Step 2: Copying raw datasets to new file...")
            with h5py.File(dst_path, 'w') as f_dst:
                for key in f_src.keys():
                    if key != 'features':
                        print(f"  Copying: {key} {f_src[key].shape}")
                        f_src.copy(key, f_dst)
                        
    except Exception as e:
        print(f"Fatal error reading source: {e}")
        return

    # STEP 3: WRITE FEATURES (File is closed now, so Pandas can open it safely)
    print("Step 3: Writing recovered features...")
    
    if recovered_df is not None:
        # CLEANUP: Rename 'Unnamed: 0' to 'index' if meaningful, or drop
        # If 'Unnamed: 0' looks like an ID (0,1,2...), we keep it as index
        unnamed_cols = [c for c in recovered_df.columns if "Unnamed" in str(c)]
        
        for c in unnamed_cols:
            # If it looks like a range index, we might want to keep it or just reset
            print(f"Found artifact column: {c}")
            # If you specifically want to KEEP indices that might be in here:
            # recovered_df.rename(columns={c: "original_index"}, inplace=True)
            # OR just drop it if it's garbage:
            recovered_df.drop(columns=[c], inplace=True)

        # Force valid index
        recovered_df.reset_index(drop=True, inplace=True)
        
        if len(recovered_df) != n_images:
            print(f"[Warn] Feature count ({len(recovered_df)}) != Image count ({n_images}). Resizing or padding might occur.")
            # Pad or truncate to match images to prevent app crashes
            if len(recovered_df) < n_images:
                # Pad
                extra = pd.DataFrame(index=range(n_images - len(recovered_df)), columns=recovered_df.columns)
                recovered_df = pd.concat([recovered_df, extra]).reset_index(drop=True)
            else:
                # Truncate
                recovered_df = recovered_df.iloc[:n_images]

        try:
            recovered_df.to_hdf(dst_path, key='features', mode='a', format='table', data_columns=True, complevel=5, complib='blosc')
            print(f"SUCCESS. Repaired file: {dst_path}")
        except Exception as e:
            print(f"Error writing features: {e}")
            
    else:
        print("Creating EMPTY features table (placeholder).")
        df_empty = pd.DataFrame({"placeholder": np.zeros(n_images)})
        df_empty.to_hdf(dst_path, key='features', mode='a', format='table', data_columns=True)
        print(f"SUCCESS (Empty Features). Repaired file: {dst_path}")

if __name__ == "__main__":
    # Update this path if needed
    file_to_fix = "/mnt/deepstore/Vidur/Junk_Classification/data/unannotated/MM_cluster_7.hdf5"
    
    if os.path.exists(file_to_fix):
        repair_file(file_to_fix)
    else:
        print(f"File not found: {file_to_fix}")