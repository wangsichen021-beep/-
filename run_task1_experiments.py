from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    run_name: str
    arch: str
    pretrained: bool
    lr_backbone: float
    lr_head: float


EXPERIMENTS = [
    Experiment("baseline_resnet18_imagenet_lr1e-4_1e-3", "resnet18", True, 1e-4, 1e-3),
    Experiment("baseline_resnet18_imagenet_lr5e-5_5e-4", "resnet18", True, 5e-5, 5e-4),
    Experiment("baseline_resnet18_imagenet_lr3e-4_3e-3", "resnet18", True, 3e-4, 3e-3),
    Experiment("ablation_resnet18_scratch_lr1e-3", "resnet18", False, 1e-3, 1e-3),
    Experiment("attention_se_resnet18_imagenet_lr1e-4_1e-3", "se_resnet18", True, 1e-4, 1e-3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 1 experiment suite.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="runs/task1")
    parser.add_argument("--data-root", default="data/oxford-iiit-pet")
    parser.add_argument("--archive-dir", default=".")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--only", nargs="*", default=None, help="Run names to execute.")
    parser.add_argument("--quick", action="store_true", help="One-epoch, two-batch smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [experiment for experiment in EXPERIMENTS if experiment.run_name in wanted]
        missing = wanted - {experiment.run_name for experiment in selected}
        if missing:
            raise SystemExit(f"Unknown experiment(s): {', '.join(sorted(missing))}")

    for experiment in selected:
        command = [
            sys.executable,
            "train_task1.py",
            "--run-name",
            experiment.run_name,
            "--arch",
            experiment.arch,
            "--epochs",
            str(1 if args.quick else args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--output-dir",
            args.output_dir,
            "--data-root",
            args.data_root,
            "--archive-dir",
            args.archive_dir,
            "--device",
            args.device,
            "--lr-backbone",
            str(experiment.lr_backbone),
            "--lr-head",
            str(experiment.lr_head),
        ]
        command.append("--pretrained" if experiment.pretrained else "--no-pretrained")
        if args.quick:
            command.extend(["--limit-train-batches", "2", "--limit-eval-batches", "2"])

        print("=" * 80, flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
