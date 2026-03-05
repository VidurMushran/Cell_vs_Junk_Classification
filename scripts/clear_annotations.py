#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from junk_gui_app.app.data.h5io import write_features_column_inplace  # noqa: E402

def main():
    ap = argparse.ArgumentParser(description="Clear / reset label annotations in HDF5 features tables.")
    ap.add_argument("--paths", nargs="+", required=True, help="One or more .hdf5 files")
    ap.add_argument("--features_key", default="features")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--set", default="nan", choices=["nan", "0", "1"], help="Set all labels to NaN, 0, or 1")
    args = ap.parse_args()

    for p in args.paths:
        p = str(Path(p))
        df = pd.read_hdf(p, key=args.features_key)
        n = len(df)

        if args.set == "nan":
            vals = np.full(n, np.nan, dtype=np.float32)
        elif args.set == "0":
            vals = np.zeros(n, dtype=np.float32)
        else:
            vals = np.ones(n, dtype=np.float32)

        write_features_column_inplace(p, args.label_col, vals, features_key=args.features_key)
        print(f"[ok] {p}: set {args.label_col} -> {args.set} (n={n})")


if __name__ == "__main__":
    main()
