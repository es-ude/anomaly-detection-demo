from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ("good", "damaged")

type Transformation = Callable[[torch.Tensor], torch.Tensor]


class CookieClfDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        training_set: bool,
        sample_transform: Optional[Transformation] = None,
        target_transform: Optional[Transformation] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.training_set = training_set
        self.sample_transform = sample_transform
        self.target_transform = target_transform
        self.samples, self.targets = self._load_dataset()

        if self.sample_transform is not None:
            self.samples = self.sample_transform(self.samples)

        if self.target_transform is not None:
            self.targets = self.target_transform(self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index], self.targets[index]

    @property
    def _base_dir(self) -> Path:
        if self.training_set:
            return self.dataset_dir / "training/v1"
        return self.dataset_dir / "testing"

    def _load_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        samples_per_class = []
        targets_per_class = []

        for class_idx, class_name in enumerate(CLASSES):
            class_dir = self._base_dir / class_name

            samples = torch.stack([_load_image(img) for img in class_dir.glob("*.jpg")])
            targets = torch.ones(len(samples), dtype=torch.long) * class_idx

            samples_per_class.append(samples)
            targets_per_class.append(targets)

        samples = torch.cat(samples_per_class)
        targets = torch.cat(targets_per_class)

        return samples, targets


def _load_image(file: Path) -> torch.Tensor:
    with Image.open(file, "r") as img:
        image = torch.tensor(np.array(img), dtype=torch.uint8)
    return image.permute(2, 0, 1)
