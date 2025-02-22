import torch
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage


class ImagePreprocessing(Compose):
    def __init__(self, target_img_width: int, target_img_height: int) -> None:
        super().__init__(
            [
                ToImage(),
                Resize((target_img_width, target_img_height)),
                ToDtype(dtype=torch.uint8, scale=True),
            ]
        )
