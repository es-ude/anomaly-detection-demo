import torch
from torchvision.transforms.v2 import (
    Compose,
    Grayscale,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToDtype,
    ToImage,
    Transform,
)


class _BasePreprocessing(Compose):
    def __init__(
        self,
        target_img_width: int,
        target_img_height: int,
        augmentations: list[Transform],
    ) -> None:
        super().__init__(
            [
                ToImage(),
                Grayscale(num_output_channels=1),
                Resize((target_img_width, target_img_height)),
                *augmentations,
                ToDtype(dtype=torch.float32, scale=True),
            ]
        )


class TrainingPreprocessing(_BasePreprocessing):
    def __init__(
        self,
        target_img_width: int,
        target_img_height: int,
    ) -> None:
        super().__init__(
            target_img_width,
            target_img_height,
            [
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ],
        )


class InferencePreprocessing(_BasePreprocessing):
    def __init__(self, target_img_width: int, target_img_height: int) -> None:
        super().__init__(target_img_width, target_img_height, [])
