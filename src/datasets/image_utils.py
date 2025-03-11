from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.v2.functional import pil_to_tensor


def get_image_paths(root_dir: Path, img_type: str) -> list[Path]:
    return list(img for img in root_dir.glob(f"*.{img_type}") if img.is_file())


def load_image(image_file: Path) -> torch.Tensor:
    with Image.open(image_file, "r") as opened_image:
        image = pil_to_tensor(opened_image.copy())
    return image
