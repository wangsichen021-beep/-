from __future__ import annotations

import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


NUM_CLASSES = 37


@dataclass(frozen=True)
class PetSample:
    image_id: str
    class_id: int
    species_id: int
    breed_id: int

    @property
    def label(self) -> int:
        return self.class_id - 1


def ensure_dataset(data_root: str | Path, archive_dir: str | Path = ".") -> Path:
    """Extract the Oxford-IIIT Pet archives if the image/annotation folders are absent."""
    data_root = Path(data_root)
    archive_dir = Path(archive_dir)
    images_dir = data_root / "images"
    annotations_dir = data_root / "annotations"

    data_root.mkdir(parents=True, exist_ok=True)
    archives = [
        (archive_dir / "images.tar.gz", images_dir),
        (archive_dir / "annotations.tar.gz", annotations_dir),
    ]
    for archive_path, expected_dir in archives:
        if expected_dir.exists():
            continue
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Missing {archive_path}. Put images.tar.gz and annotations.tar.gz in {archive_dir}."
            )
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_root)

    return data_root


def read_split_file(data_root: str | Path, split_name: str) -> list[PetSample]:
    split_path = Path(data_root) / "annotations" / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    samples: list[PetSample] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, class_id, species_id, breed_id = line.split()
            samples.append(
                PetSample(
                    image_id=image_id,
                    class_id=int(class_id),
                    species_id=int(species_id),
                    breed_id=int(breed_id),
                )
            )
    return samples


def stratified_train_val_split(
    samples: list[PetSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[PetSample], list[PetSample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    by_class: dict[int, list[PetSample]] = {}
    for sample in samples:
        by_class.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    train_samples: list[PetSample] = []
    val_samples: list[PetSample] = []
    for class_samples in by_class.values():
        shuffled = class_samples[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def load_samples(
    data_root: str | Path,
    split: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[PetSample]:
    if split == "trainval":
        return read_split_file(data_root, "trainval")
    if split == "test":
        return read_split_file(data_root, "test")
    if split in {"train", "val"}:
        trainval = read_split_file(data_root, "trainval")
        train_samples, val_samples = stratified_train_val_split(trainval, val_ratio, seed)
        return train_samples if split == "train" else val_samples
    raise ValueError(f"Unsupported split '{split}'. Use train, val, trainval, or test.")


class OxfordIIITPetClassification(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        transform: Callable | None = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.transform = transform
        self.samples = load_samples(self.data_root, split, val_ratio, seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.images_dir / f"{sample.image_id}.jpg"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label
