from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation accuracy curves for all Task 1 runs.")
    parser.add_argument("--runs-dir", default="runs/task1_submit")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    plt.figure(figsize=(10, 5.5))

    for metrics_path in sorted(runs_dir.glob("*/metrics.csv")):
        rows = read_metrics(metrics_path)
        epochs = [row["epoch"] for row in rows]
        val_acc = [row["val_acc"] for row in rows]
        plt.plot(epochs, val_acc, marker="o", linewidth=2, label=metrics_path.parent.name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = Path(args.output) if args.output else runs_dir / "val_accuracy_curves.png"
    plt.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
