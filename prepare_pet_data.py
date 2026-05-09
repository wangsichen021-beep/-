from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pet_data import NUM_CLASSES, ensure_dataset, load_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Oxford-IIIT Pet Dataset archives.")
    parser.add_argument("--data-root", default="data/oxford-iiit-pet", help="Extracted dataset directory.")
    parser.add_argument("--archive-dir", default=".", help="Directory containing images.tar.gz and annotations.tar.gz.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = ensure_dataset(args.data_root, args.archive_dir)

    for split in ["train", "val", "trainval", "test"]:
        samples = load_samples(data_root, split, args.val_ratio, args.seed)
        labels = Counter(sample.label for sample in samples)
        print(f"{split:8s}: {len(samples):4d} images, {len(labels):2d}/{NUM_CLASSES} classes")

    images_dir = Path(data_root) / "images"
    annotations_dir = Path(data_root) / "annotations"
    print(f"images      : {images_dir.resolve()}")
    print(f"annotations : {annotations_dir.resolve()}")


if __name__ == "__main__":
    main()
