from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Task 1 summaries into a CSV and plot.")
    parser.add_argument("--runs-dir", default="runs/task1")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-plot", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    summaries = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8-sig") as f:
            summaries.append(json.load(f))

    if not summaries:
        raise SystemExit(f"No summary.json files found under {runs_dir}.")

    summaries.sort(key=lambda item: (item["test_acc_at_best"], item["best_val_acc"]), reverse=True)
    fieldnames = [
        "run_name",
        "arch",
        "pretrained",
        "epochs",
        "best_epoch",
        "best_val_acc",
        "test_acc_at_best",
        "final_test_acc",
        "total_seconds",
        "run_dir",
    ]
    output_csv = Path(args.output_csv) if args.output_csv else runs_dir / "summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})

    labels = [row["run_name"] for row in summaries]
    values = [row["test_acc_at_best"] for row in summaries]
    plt.figure(figsize=(max(9, len(labels) * 1.8), 5))
    bars = plt.bar(range(len(labels)), values, color="#3274a1")
    plt.ylabel("Test Accuracy at Best Val")
    plt.ylim(0, 1)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()

    output_plot = Path(args.output_plot) if args.output_plot else runs_dir / "summary_accuracy.png"
    plt.savefig(output_plot, dpi=160)

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_plot}")
    print("Best runs:")
    for row in summaries:
        print(
            f"{row['test_acc_at_best']:.4f} test | {row['best_val_acc']:.4f} val | "
            f"{row['run_name']}"
        )


if __name__ == "__main__":
    main()
