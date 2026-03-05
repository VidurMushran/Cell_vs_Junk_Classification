import argparse
import json


def load_metrics(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Compare CV accuracy/consistency between runs with and without augmentation."
    )
    ap.add_argument("--aug_metrics", type=str, required=True,
                    help="Path to cv_metrics_*.json for the augmented run.")
    ap.add_argument("--noaug_metrics", type=str, required=True,
                    help="Path to cv_metrics_*.json for the non-augmented run.")
    args = ap.parse_args()

    aug = load_metrics(args.aug_metrics)
    noaug = load_metrics(args.noaug_metrics)

    def summarize(name, m):
        agg = m["aggregate"]
        print(f"\n{name} ({m['run_name']}):")
        print(
            f"  acc={agg['mean']['acc']:.4f}±{agg['std']['acc']:.4f}, "
            f"auc={agg['mean']['auc']:.4f}±{agg['std']['auc']:.4f}, "
            f"f1={agg['mean']['f1']:.4f}±{agg['std']['f1']:.4f}"
        )

    summarize("Augmented", aug)
    summarize("No-Aug", noaug)

    diff_acc = aug["aggregate"]["mean"]["acc"] - noaug["aggregate"]["mean"]["acc"]
    diff_auc = aug["aggregate"]["mean"]["auc"] - noaug["aggregate"]["mean"]["auc"]
    diff_f1 = aug["aggregate"]["mean"]["f1"] - noaug["aggregate"]["mean"]["f1"]

    print("\nDifference (Aug - NoAug):")
    print(f"  Δacc={diff_acc:.4f}, Δauc={diff_auc:.4f}, Δf1={diff_f1:.4f}")

    print("\nConsistency (CV std):" )
    print(
        f"  Aug std acc={aug['aggregate']['std']['acc']:.4f}, "
        f"NoAug std acc={noaug['aggregate']['std']['acc']:.4f}"
    )
    print(
        f"  Aug std f1={aug['aggregate']['std']['f1']:.4f}, "
        f"NoAug std f1={noaug['aggregate']['std']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
