import torch
from torchvision.transforms.v2 import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToDtype,
    ToImage,
)


class ImagePreprocessing(Compose):
    def __init__(
        self,
        target_img_width: int,
        target_img_height: int,
        augment_images: bool = False,
    ) -> None:
        if augment_images:
            augmentations = [
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        else:
            augmentations = []

        super().__init__(
            [
                ToImage(),
                Resize((target_img_width, target_img_height)),
                *augmentations,
                ToDtype(dtype=torch.float32, scale=True),
            ]
        )
