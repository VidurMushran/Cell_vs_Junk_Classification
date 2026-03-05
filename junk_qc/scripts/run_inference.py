import argparse
import torch

from junk_qc.inference.infer_and_persist import run_inference_on_subfolder


def main():
    ap = argparse.ArgumentParser(description="Run inference with a trained checkpoint and persist predictions.")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a .pt checkpoint produced by main_train/run_cv_training.")
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory containing a subfolder of HDF5 files.")
    ap.add_argument("--subfolder", type=str, default="unannotated",
                    help="Subfolder inside root (default: unannotated)."),
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    run_inference_on_subfolder(args.checkpoint, root=args.root, subfolder=args.subfolder, device=device)


if __name__ == "__main__":
    main()
