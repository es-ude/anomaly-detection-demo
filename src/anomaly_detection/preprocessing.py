import torch
import torchvision.transforms.v2 as transforms


class _BasePreprocessing(transforms.Compose):
    def __init__(
        self,
        target_img_height: int,
        target_img_width: int,
        augmentations: list[transforms.Transform],
    ) -> None:
        super().__init__(
            [
                transforms.ToImage(),
                transforms.Grayscale(num_output_channels=1),
                # transforms.functional.autocontrast,
                transforms.Resize((target_img_height, target_img_width)),
                *augmentations,
                transforms.ToDtype(dtype=torch.float32, scale=True),
            ]
        )


class TrainingPreprocessing(_BasePreprocessing):
    def __init__(self, target_img_height: int, target_img_width: int) -> None:
        super().__init__(
            target_img_height,
            target_img_width,
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ],
        )


class InferencePreprocessing(_BasePreprocessing):
    def __init__(self, target_img_height: int, target_img_width: int) -> None:
        super().__init__(target_img_height, target_img_width, [])
