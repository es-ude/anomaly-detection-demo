import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, Protocol

import cv2

from demo.camera.image import Image, convert_rgb_to_bgr

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


class _Camera(Protocol):
    def __init__(self, cam_port: int | str, width: int, height: int): ...
    def is_opened(self) -> bool: ...
    def read_frame(self) -> Image | None: ...
    def release(self) -> None: ...


def _get_camera(camera: Literal["opencv", "picam"], port: int) -> _Camera:
    match camera:
        case "opencv":
            from demo.camera.opencv_camera import Camera
        case "picam":
            from demo.camera.picamv3_camera import Camera
    return Camera(cam_port=port, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)


def _get_image_name() -> str:
    return f"{uuid.uuid4().hex}.jpg"


def _save_image(image: Image, destination: Path) -> None:
    cv2.imwrite(
        filename=str(destination),
        img=convert_rgb_to_bgr(image),
    )


def main(args: Namespace) -> None:
    camera = _get_camera(camera=args.camera, port=args.camera_port)

    img_dir: Path = args.image_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        while camera.is_opened():
            _ = input("Press 'enter' to take a picture...")

            image = camera.read_frame()

            if image is None:
                print("[Error] Could not take picture.")
            else:
                _save_image(image, destination=img_dir / _get_image_name())
    except KeyboardInterrupt:
        return
    finally:
        camera.release()

    print("[Error] Camera closed.")


if __name__ == "__main__":
    argparser = ArgumentParser("Dataset Recorder")
    argparser.add_argument("-d", "--image-dir", type=Path)
    argparser.add_argument(
        "-c", "--camera", default="picam", choices=("opencv", "picam")
    )
    argparser.add_argument("-p", "--camera-port", default=0, type=int)
    args = argparser.parse_args()

    main(args)
