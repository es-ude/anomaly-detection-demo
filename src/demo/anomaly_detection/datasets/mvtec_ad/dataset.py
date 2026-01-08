from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToImage
from torchvision.tv_tensors import Image

from demo.anomaly_detection.datasets.image_utils import get_image_paths, load_image

from .constants import CLASSES


class MVTecAdDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        object: str,
        training_set: bool,
        anomalies: Optional[list[str]] = None,
        sample_transform: Optional[Callable[[Image], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        in_memory: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.object = object
        self.training_set = training_set
        self.anomalies = CLASSES[self.object] if anomalies is None else anomalies
        self.sample_transform = (
            ToImage()
            if sample_transform is None
            else Compose([ToImage(), sample_transform])
        )
        self.target_transform = target_transform
        self.in_memory = in_memory

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
            len(get_image_paths(cls_dir, "png"))
            for cls_dir in self._get_anomaly_class_dirs()
        )

    def _determine_image_label_pairs(self) -> list[tuple[Path, torch.Tensor]]:
        def label(obj_name: str, anomaly: str) -> torch.Tensor:
            return torch.tensor(CLASSES[obj_name].index(anomaly))

        return [
            (img, label(self.object, cls_dir.name))
            for cls_dir in self._get_anomaly_class_dirs()
            for img in get_image_paths(cls_dir, "png")
        ]

    def _get_anomaly_class_dirs(self) -> list[Path]:
        split_name = "train" if self.training_set else "test"
        object_split_dir = self.dataset_dir / self.object / split_name
        return [object_split_dir / anomaly for anomaly in self.anomalies]

    def _load_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = [], []
        for image_file, label in self._image_label_pairs:
            image = load_image(image_file)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.stack(labels)
