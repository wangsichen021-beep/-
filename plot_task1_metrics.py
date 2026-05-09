from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one Task 1 run's learning curves.")
    parser.add_argument("run_dir", help="Directory containing metrics.csv.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = read_metrics(run_dir / "metrics.csv")
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_acc"] for row in rows], marker="o", label="train")
    axes[0].plot(epochs, [row["val_acc"] for row in rows], marker="o", label="val")
    axes[0].plot(epochs, [row["test_acc"] for row in rows], marker="o", label="test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train")
    axes[1].plot(epochs, [row["val_loss"] for row in rows], marker="o", label="val")
    axes[1].plot(epochs, [row["test_loss"] for row in rows], marker="o", label="test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(args.output) if args.output else run_dir / "learning_curves.png"
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
