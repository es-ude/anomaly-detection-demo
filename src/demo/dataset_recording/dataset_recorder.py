import time
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, Protocol

from demo.camera.image import Image
from demo.dataset_recording.utils import IMG_EXT, save_image


class _Camera(Protocol):
    def __init__(self, cam_port: int | str, width: int, height: int): ...
    def is_opened(self) -> bool: ...
    def read_frame(self) -> Image | None: ...
    def release(self) -> None: ...


def _get_camera(
    camera: Literal["opencv", "picam"], port: int, image_width: int, image_height: int
) -> _Camera:
    match camera:
        case "opencv":
            from demo.camera.opencv_camera import Camera
        case "picam":
            from demo.camera.picamv3_camera import Camera
    return Camera(cam_port=port, width=image_width, height=image_height)


def _wait_to_capture_new_image(delay: int, alert_before_sleep: bool) -> None:
    if args.delay < 0:
        _ = input("[*] Press 'enter' to take a picture...")
    else:
        if alert_before_sleep:
            print("\a", end="")

        time.sleep(delay)


def _get_image_name() -> str:
    return f"{uuid.uuid4().hex}.{IMG_EXT}"


def main(args: Namespace) -> None:
    camera = _get_camera(
        camera=args.camera,
        port=args.camera_port,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    img_dir: Path = args.image_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        while camera.is_opened():
            _wait_to_capture_new_image(args.delay, alert_before_sleep=args.alert)

            image = camera.read_frame()
            print("[+] New image captured.")

            if image is None:
                print("[!] Could not take picture.")
            else:
                save_image(image, destination=img_dir / _get_image_name())
    except KeyboardInterrupt:
        return
    finally:
        camera.release()

    print("[!] Camera closed.")


if __name__ == "__main__":
    argparser = ArgumentParser("Dataset Recorder")
    argparser.add_argument("-d", "--image-dir", type=Path)
    argparser.add_argument(
        "-c", "--camera", default="picam", choices=("opencv", "picam")
    )
    argparser.add_argument("-p", "--camera-port", default=0, type=int)
    argparser.add_argument("-t", "--delay", default=-1, type=int)
    argparser.add_argument("--alert", default=True, type=bool)
    argparser.add_argument("--image-width", default=1920, type=int)
    argparser.add_argument("--image-height", default=1080, type=int)
    args = argparser.parse_args()

    main(args)
