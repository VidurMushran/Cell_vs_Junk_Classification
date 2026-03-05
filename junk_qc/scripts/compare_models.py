import argparse
import json


def load_metrics(path: str):
    with open(path, "r") as f:
        return json.load(f)


def print_summary(label: str, m):
    agg = m["aggregate"]
    print(f"\n{label} ({m['run_name']}):")
    print(
        f"  acc={agg['mean']['acc']:.4f}±{agg['std']['acc']:.4f}, "
        f"auc={agg['mean']['auc']:.4f}±{agg['std']['auc']:.4f}, "
        f"f1={agg['mean']['f1']:.4f}±{agg['std']['f1']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare consistency and accuracy of simple CNN vs rotation-invariant CNN "
            "vs ImageNet-pretrained backbone.")
    )
    ap.add_argument("--simple", type=str, required=True,
                    help="Path to cv_metrics_*.json for the simple CNN run.")
    ap.add_argument("--rot", type=str, required=True,
                    help="Path to cv_metrics_*.json for the rotation-invariant CNN run.")
    ap.add_argument("--pretrained", type=str, required=True,
                    help="Path to cv_metrics_*.json for the ImageNet-pretrained run.")
    args = ap.parse_args()

    simple_m = load_metrics(args.simple)
    rot_m = load_metrics(args.rot)
    pre_m = load_metrics(args.pretrained)

    print_summary("Simple CNN", simple_m)
    print_summary("Rotation-invariant CNN", rot_m)
    print_summary("Pretrained (last layers unfrozen)", pre_m)

    print("\nRelative differences (using Simple CNN as baseline):")
    def diff_line(name, base, other):
        d_acc = other["aggregate"]["mean"]["acc"] - base["aggregate"]["mean"]["acc"]
        d_auc = other["aggregate"]["mean"]["auc"] - base["aggregate"]["mean"]["auc"]
        d_f1 = other["aggregate"]["mean"]["f1"] - base["aggregate"]["mean"]["f1"]
        print(f"  {name}: Δacc={d_acc:.4f}, Δauc={d_auc:.4f}, Δf1={d_f1:.4f}")

    diff_line("Rotation-invariant", simple_m, rot_m)
    diff_line("Pretrained", simple_m, pre_m)

    print("\nConsistency comparison (CV std):")
    for label, m in [("Simple", simple_m), ("Rot", rot_m), ("Pre", pre_m)]:
        print(
            f"  {label}: std acc={m['aggregate']['std']['acc']:.4f}, "
            f"std f1={m['aggregate']['std']['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
