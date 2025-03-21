import torch
import torchvision.transforms.v2 as transforms


class _BasePreprocessing(transforms.Compose):
    def __init__(
        self,
        target_img_width: int,
        target_img_height: int,
        augmentations: list[transforms.Transform],
    ) -> None:
        super().__init__(
            [
                transforms.ToImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.functional.equalize,
                transforms.Resize((target_img_width, target_img_height)),
                *augmentations,
                transforms.ToDtype(dtype=torch.float32, scale=True),
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ],
        )


class InferencePreprocessing(_BasePreprocessing):
    def __init__(self, target_img_width: int, target_img_height: int) -> None:
        super().__init__(target_img_width, target_img_height, [])
