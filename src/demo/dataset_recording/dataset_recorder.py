import time
import uuid
from pathlib import Path
from typing import Literal, Protocol

import typer

from demo.camera.image import Image
from demo.dataset_recording.utils import IMG_EXT, save_image


def main(
    image_dir: Path,
    camera_backend: Literal["opencv", "picam"] = "opencv",
    camera_port: int = 0,
    delay: int = -1,
    alert: bool = True,
    image_width: int = 1920,
    image_height: int = 1080,
    lens_position: None | float = None,
) -> None:
    camera = _get_camera(
        camera=camera_backend,
        port=camera_port,
        image_width=image_width,
        image_height=image_height,
        lens_position=lens_position,
    )

    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        while camera.is_opened():
            _wait_to_capture_new_image(delay, alert_before_sleep=alert)

            image = camera.read_frame()
            print("[+] New image captured.")

            if image is None:
                print("[!] Could not take picture.")
            else:
                save_image(image, destination=image_dir / _get_image_name())
    except KeyboardInterrupt:
        return
    finally:
        camera.release()

    print("[!] Camera closed.")


class _Camera(Protocol):
    def __init__(self, cam_port: int | str, width: int, height: int): ...
    def is_opened(self) -> bool: ...
    def read_frame(self) -> Image | None: ...
    def release(self) -> None: ...


def _get_camera(
    camera: Literal["opencv", "picam"],
    port: int,
    image_width: int,
    image_height: int,
    lens_position: None | float,
) -> _Camera:
    match camera:
        case "opencv":
            from demo.camera.opencv_camera import Camera
        case "picam":
            from demo.camera.picamv3_camera import Camera
    return Camera(
        cam_port=port,
        width=image_width,
        height=image_height,
        lens_position=lens_position,
    )


def _wait_to_capture_new_image(delay: int, alert_before_sleep: bool) -> None:
    if delay < 0:
        _ = input("[*] Press 'enter' to take a picture...")
    else:
        if alert_before_sleep:
            print("\a", end="")

        time.sleep(delay)


def _get_image_name() -> str:
    return f"{uuid.uuid4().hex}.{IMG_EXT}"


if __name__ == "__main__":
    typer.run(main)
