from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

SampleType = Image.Image | torch.Tensor
LabelType = int | torch.Tensor


class MVTecAD(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        object: str,
        training_set: bool,
        sample_transform: Optional[Callable[[Image.Image], SampleType]] = None,
        target_transform: Optional[Callable[[int], LabelType]] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.object = object
        self.training_set = training_set
        self.sample_transfrom = sample_transform
        self.target_transfrom = target_transform
        self._dataset_len = self._determine_dataset_len()
        self._image_label_pairs = self._determine_image_label_pairs()

    def __len__(self) -> int:
        return self._dataset_len

    def __getitem__(self, index: int) -> tuple[SampleType, LabelType]:
        image_file, label = self._image_label_pairs[index]

        with Image.open(image_file, "r") as opened_image:
            image = opened_image.copy()

        if self.sample_transfrom is not None:
            image = self.sample_transfrom(image)

        if self.target_transfrom is not None:
            label = self.target_transfrom(label)

        return image, label

    def _determine_dataset_len(self) -> int:
        return sum(
            len(_get_images(cls_dir)) for cls_dir in self._get_anomaly_class_dirs()
        )

    def _determine_image_label_pairs(self) -> list[tuple[Path, int]]:
        class_dirs = self._get_anomaly_class_dirs()

        good_class_dir = class_dirs[0]
        for cls_dir in class_dirs:
            if cls_dir.name == "good":
                good_class_dir = cls_dir

        class_dirs.remove(good_class_dir)
        class_dirs.sort(key=lambda p: p.name)
        class_dirs.insert(0, good_class_dir)

        return [
            (img, cls_idx)
            for cls_idx, cls_dir in enumerate(class_dirs)
            for img in _get_images(cls_dir)
        ]

    def _get_anomaly_class_dirs(self) -> list[Path]:
        split_name = "train" if self.training_set else "test"
        object_split_dir = self.dataset_dir / self.object / split_name
        return [p for p in object_split_dir.glob("*") if p.is_dir()]


def _get_images(dir: Path) -> list[Path]:
    return list(img for img in dir.glob("*.png") if img.is_file())
