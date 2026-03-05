import pandas as pd
import h5py
from pathlib import Path
import sys


# ------------------------------
# Canonical schema normalization
# ------------------------------

REQUIRED_COLUMNS = [
    "slide_id",
    "frame_id",
    "cell_id",
    "cell_x",
    "cell_y",
    "image_id",
]

COLUMN_ALIASES = {
    "x": "cell_x",
    "cellx": "cell_x",
    "cell_x": "cell_x",
    "y": "cell_y",
    "celly": "cell_y",
    "cell_y": "cell_y",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    # Strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Drop ANY Unnamed columns (case insensitive)
    df = df.drop(
        columns=[c for c in df.columns if c.lower().startswith("unnamed")],
        errors="ignore"
    )

    # Standardize case for alias matching
    rename_map = {}
    for c in df.columns:
        cl = c.lower()

        if cl in ["x", "cellx"]:
            rename_map[c] = "cell_x"
        elif cl in ["y", "celly"]:
            rename_map[c] = "cell_y"
        elif cl == "cell_x":
            rename_map[c] = "cell_x"
        elif cl == "cell_y":
            rename_map[c] = "cell_y"

    df = df.rename(columns=rename_map)

    return df


def repair(h5_path: str):
    h5_path = Path(h5_path)
    manifest_path = h5_path.with_suffix(".manifest.parquet")

    if not manifest_path.exists():
        print("No manifest found.")
        return

    print(f"Repairing {h5_path.name}")
    manifest = pd.read_parquet(manifest_path)

    if "status" not in manifest.columns:
        print("[ERROR] Manifest missing 'status' column.")
        return

    included = manifest[manifest.status == "included"].copy()

    if included.empty:
        print("No included rows.")
        return

    included = normalize_columns(included)

    if "hdf5_index" not in included.columns:
        print("[ERROR] Missing hdf5_index in manifest.")
        return

    included = included.sort_values("hdf5_index").reset_index(drop=True)

    # Assign image_id from hdf5_index
    included["image_id"] = included["hdf5_index"]

    included["__source_h5__"] = h5_path.name

    # Validate required columns
    for col in REQUIRED_COLUMNS:
        if col not in included.columns:
            print(f"[WARNING] Required column missing: {col}")

    # Validate image count
    with h5py.File(h5_path, "r") as f:
        n_images = f["images"].shape[0]

    if len(included) != n_images:
        print(f"[WARNING] Feature rows ({len(included)}) != image count ({n_images})")

    print("Writing normalized features table...")

    with pd.HDFStore(h5_path, mode="a") as store:
        if "features" in store:
            store.remove("features")

        store.put(
            "features",
            included,
            format="table",
            data_columns=True,
            index=False
        )

    print("Repair complete.\n")


if __name__ == "__main__":
    repair(sys.argv[1])
