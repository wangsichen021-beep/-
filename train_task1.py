from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import (
    MODEL_NAMES,
    build_model,
    count_trainable_parameters,
    describe_parameter_groups,
    split_parameter_groups,
)
from pet_data import NUM_CLASSES, OxfordIIITPetClassification, ensure_dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNNs on Oxford-IIIT Pet classification.")
    parser.add_argument("--data-root", default="data/oxford-iiit-pet")
    parser.add_argument("--archive-dir", default=".")
    parser.add_argument("--output-dir", default="runs/task1")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--arch", choices=MODEL_NAMES, default="resnet18")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument(
        "--test-every",
        type=int,
        default=0,
        help="Evaluate test set every N epochs. Default 0 tests only once using the best val checkpoint.",
    )
    parser.add_argument("--progress", action="store_true", help="Force tqdm progress bars.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_root = ensure_dataset(args.data_root, args.archive_dir)
    train_transform, eval_transform = build_transforms(args.image_size)
    train_set = OxfordIIITPetClassification(
        data_root, "train", train_transform, args.val_ratio, args.seed
    )
    val_set = OxfordIIITPetClassification(data_root, "val", eval_transform, args.val_ratio, args.seed)
    test_set = OxfordIIITPetClassification(data_root, "test", eval_transform, args.val_ratio, args.seed)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    return int((preds == targets).sum().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    limit_batches: int | None,
    show_progress: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc="train", leave=False, disable=not show_progress)
    for batch_idx, (images, targets) in enumerate(progress, start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy(logits.detach(), targets)
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / total_examples, acc=total_correct / total_examples)

    return {
        "loss": total_loss / max(1, total_examples),
        "acc": total_correct / max(1, total_examples),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    split_name: str,
    limit_batches: int | None,
    show_progress: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=split_name, leave=False, disable=not show_progress)
    for batch_idx, (images, targets) in enumerate(progress, start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy(logits, targets)
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / total_examples, acc=total_correct / total_examples)

    return {
        "loss": total_loss / max(1, total_examples),
        "acc": total_correct / max(1, total_examples),
    }


def default_run_name(args: argparse.Namespace) -> str:
    pretrain_tag = "imagenet" if args.pretrained else "scratch"
    return (
        f"{args.arch}_{pretrain_tag}_e{args.epochs}"
        f"_blr{args.lr_backbone:g}_hlr{args.lr_head:g}_seed{args.seed}"
    )


def save_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def finite_or_none(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else value


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    history: list[dict],
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    show_progress = bool(args.progress or sys.stderr.isatty())

    train_loader, val_loader, test_loader = make_loaders(args)
    model = build_model(args.arch, NUM_CLASSES, args.pretrained).to(device)
    param_groups = split_parameter_groups(model, args.lr_backbone, args.lr_head, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = args.run_name or default_run_name(args)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "run_name": run_name,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "num_classes": NUM_CLASSES,
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
            "trainable_parameters": count_trainable_parameters(model),
            "parameter_groups": describe_parameter_groups(param_groups),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Run       : {run_name}")
    print(f"Device    : {device}")
    if device.type == "cuda":
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"Train/Val/Test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    history: list[dict] = []
    best_val_acc = -math.inf
    best_epoch = 0
    test_at_best = {"loss": math.inf, "acc": 0.0}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            args.limit_train_batches,
            show_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            "val",
            args.limit_eval_batches,
            show_progress,
        )
        should_test = args.test_every > 0 and epoch % args.test_every == 0
        test_metrics = {"loss": math.nan, "acc": math.nan}
        if should_test:
            test_metrics = evaluate(
                model,
                test_loader,
                criterion,
                device,
                use_amp,
                "test",
                args.limit_eval_batches,
                show_progress,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        save_history_csv(run_dir / "metrics.csv", history)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
            f"test_acc={row['test_acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            torch.save(
                checkpoint_payload(model, optimizer, epoch, args, history),
                run_dir / "best.pt",
            )

        torch.save(
            checkpoint_payload(model, optimizer, epoch, args, history),
            run_dir / "last.pt",
        )

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_at_best = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_amp,
        "test",
        args.limit_eval_batches,
        show_progress,
    )
    for row in history:
        if int(row["epoch"]) == best_epoch:
            row["test_loss"] = test_at_best["loss"]
            row["test_acc"] = test_at_best["acc"]
    save_history_csv(run_dir / "metrics.csv", history)

    summary = {
        "run_name": run_name,
        "arch": args.arch,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc_at_best": test_at_best["acc"],
        "test_loss_at_best": test_at_best["loss"],
        "final_val_acc": history[-1]["val_acc"],
        "final_test_acc": finite_or_none(history[-1]["test_acc"]),
        "total_seconds": time.time() - start_time,
        "run_dir": str(run_dir),
    }
    summary_text = json.dumps(summary, indent=2, allow_nan=False)
    (run_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
