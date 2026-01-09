from pathlib import Path

import cv2

from demo.camera.image import Image, convert_bgr_to_rgb, convert_rgb_to_bgr

IMG_EXT = "jpg"


def load_image(image_file: Path) -> Image:
    image = cv2.imread(filename=str(image_file))
    return convert_bgr_to_rgb(image)


def save_image(image: Image, destination: Path) -> None:
    cv2.imwrite(
        filename=str(destination),
        img=convert_rgb_to_bgr(image),
    )
