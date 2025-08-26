from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToImage
from torchvision.transforms.v2.functional import crop
from torchvision.tv_tensors import Image

from src.anomaly_detection.datasets.image_utils import get_image_paths, load_image

CLASSES = dict(train=["good"], test=["good", "bad"])
_AREA_TO_CROP = (180, 490, 900, 900)


class CookieAdDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        training_set: bool,
        dataset_version: int = 2,
        sample_transform: Optional[Callable[[Image], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        in_memory: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.training_set = training_set

        additional_transforms = []

        if dataset_version == 1:
            additional_transforms.append(lambda img: crop(img, *_AREA_TO_CROP))

        if sample_transform is not None:
            additional_transforms.append(sample_transform)

        self.sample_transform = Compose([ToImage(), *additional_transforms])
        self.target_transform = target_transform
        self.in_memory = in_memory

        self._split_name = "train" if self.training_set else "test"
        self._dataset_len = self._determine_dataset_len()
        self._image_label_pairs = self._determine_image_label_pairs()

        if self.in_memory:
            self._loaded_samples, self._loaded_targets = self._load_dataset()

    def __len__(self) -> int:
        return self._dataset_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            image, label = self._loaded_samples[index], self._loaded_targets[index]
        else:
            image_file, label = self._image_label_pairs[index]
            image = load_image(image_file)

        if self.sample_transform is not None:
            image = self.sample_transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def _determine_dataset_len(self) -> int:
        return sum(
            len(get_image_paths(cls_dir, "jpg")) for cls_dir in self._get_class_dirs()
        )

    def _determine_image_label_pairs(self) -> list[tuple[Path, torch.Tensor]]:
        def label(cls: str) -> torch.Tensor:
            return torch.tensor(CLASSES[self._split_name].index(cls))

        return [
            (img_file, label(cls_dir.name))
            for cls_dir in self._get_class_dirs()
            for img_file in get_image_paths(cls_dir, "jpg")
        ]

    def _get_class_dirs(self) -> list[Path]:
        return [
            self.dataset_dir / self._split_name / cls
            for cls in CLASSES[self._split_name]
        ]

    def _load_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = [], []
        for image_file, label in self._image_label_pairs:
            image = load_image(image_file)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.stack(labels)
