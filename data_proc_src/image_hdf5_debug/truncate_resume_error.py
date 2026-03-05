import h5py
import pandas as pd

h5 = "common_cell/common_cell:shard_05.hdf5"

# Load manifest
manifest = pd.read_parquet(h5.replace(".hdf5", ".manifest.parquet"))
included = manifest[manifest.status == "included"]

n_valid = len(included)

with h5py.File(h5, "a") as f:
    imgs = f["images"]
    print("Before resize:", imgs.shape)
    imgs.resize(n_valid, axis=0)
    print("After resize:", imgs.shape)

print("Shard_05 truncated safely.")