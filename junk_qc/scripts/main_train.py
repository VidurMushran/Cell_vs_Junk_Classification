import argparse
import time
import torch

from junk_qc.utils.seed import set_seed
from junk_qc.utils.io import ensure_dir
from junk_qc.data.index_build import build_annotated_index
from junk_qc.data.h5_dataset import estimate_channel_stats
from junk_qc.train.train_binary_cv import run_cv_training

def load_yaml_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def parse_args():
    ap = argparse.ArgumentParser(description="Train junk-vs-rest classifiers with 5-fold CV.")
    ap.add_argument("--root", type=str, required=True,
                    help="Root containing subfolders: junk_annotated, rare_cells_annotated, wbcs_annotated, unannotated.")
    ap.add_argument("--out_dir", type=str, default="pipeline_out",
                    help="Experiment output directory (default: pipeline_out -> experiment_outputs_<timestamp>)."),
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--nonjunk_to_junk", type=float, default=1.5)
    ap.add_argument("--junk_k_max", type=int, default=8)
    ap.add_argument("--junk_max_rows_for_kmeans", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_amp", action="store_true")

    # Model config
    ap.add_argument("--arch", type=str, default="simple_cnn",
                    help="Model architecture: simple_cnn, rot_invariant_cnn, vgg19_bn, resnet18.")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use ImageNet-pretrained weights when available (for torchvision backbones)."),
    ap.add_argument("--unfreeze_last", type=int, default=0,
                    help="If >0 and pretrained, unfreeze last N layers (QUALIFAI-style)."),
    ap.add_argument("--no_qualifai_aug", action="store_true",
                    help="Disable QUALIFAI-style geometric augmentation on the training set."),
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.out_dir == "pipeline_out" or not args.out_dir:
        out_dir = f"experiment_outputs_{timestamp}"
    else:
        out_dir = args.out_dir
    ensure_dir(out_dir)

    print("Building annotated index & junk clusters for training...")
    items, k_star = build_annotated_index(
        args.root,
        target_hw=75,
        junk_k_max=args.junk_k_max,
        junk_max_rows_for_kmeans=args.junk_max_rows_for_kmeans,
        seed=args.seed,
    )
    print(f"Chosen K for junk clusters: {k_star}")

    ch_mean, ch_std = estimate_channel_stats(items, max_samples=6000, target_hw=75)
    print("Channel mean:", [round(x, 2) for x in ch_mean])
    print("Channel std :", [round(x, 2) for x in ch_std])

    results = run_cv_training(
        items=items,
        ch_mean=ch_mean,
        ch_std=ch_std,
        out_dir=out_dir,
        arch=args.arch,
        pretrained=args.pretrained,
        unfreeze_last=args.unfreeze_last,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        gamma=args.gamma,
        use_amp=args.use_amp,
        device=device,
        use_qualifai_aug=not args.no_qualifai_aug,
        nonjunk_to_junk=args.nonjunk_to_junk,
        n_splits=5,
        seed=args.seed,
        timestamp=timestamp,
    )

    print("\n=== Summary ===")
    print(f"Run name      : {results['run_name']}")
    print(f"Timestamp     : {results['timestamp']}")
    print("Fold metrics  :")
    for k, v in results["folds"].items():
        print(f"  Fold {k}: acc={v['acc']:.3f}, auc={v['auc']:.3f}, f1={v['f1']:.3f}")
    agg = results["aggregate"]
    print("Aggregate CV  :")
    print(
        f"  acc={agg['mean']['acc']:.3f}±{agg['std']['acc']:.3f}, "
        f"auc={agg['mean']['auc']:.3f}±{agg['std']['auc']:.3f}, "
        f"f1={agg['mean']['f1']:.3f}±{agg['std']['f1']:.3f}"
    )
    print(f"Final model   : {results['final_model_path']}")


if __name__ == "__main__":
    main()
