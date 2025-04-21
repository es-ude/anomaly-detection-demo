import base64
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import cv2
from nicegui import run

from src.anomaly_detection.anomaly_detector import DetectionResult
from src.demo_interface.camera import Camera
from src.demo_interface.image import Image, convert_rgb_to_bgr
from src.demo_interface.image_processing import ImageProcessor

type UpdateUICallback = Callable[[dict[str, str]], None] | Callable[[str], None]


class _NoneImageProcessor:
    def process(self, image: Image | None) -> None:
        return None


class DemoApplicationController:
    def __init__(self, cam_port: int | str, placeholder_image_file: Path) -> None:
        self._camera = Camera(cam_port, width=1920, height=1080)
        self._placeholder_image = _load_image(placeholder_image_file)
        self._image_processor: ImageProcessor = _NoneImageProcessor()
        self._update_ui_callback: UpdateUICallback = lambda _: None

    async def run(self) -> None:
        while True:
            processed_frame = await self._take_and_process_frame()
            ui_data = self._frame_to_ui_data(processed_frame)
            self._update_ui_callback(ui_data)  # type: ignore

    def set_update_ui_callback(self, callback: UpdateUICallback) -> None:
        self._update_ui_callback = callback

    def set_image_processor(self, image_processor: ImageProcessor) -> None:
        self._image_processor = image_processor

    def close(self) -> None:
        self._camera.release()

    async def _take_and_process_frame(self) -> DetectionResult | Image | None:
        frame = await run.io_bound(self._camera.read_frame)
        return await run.cpu_bound(self._image_processor.process, frame)

    def _frame_to_ui_data(
        self, frame: DetectionResult | Image | None
    ) -> dict[str, str] | str:
        if frame is None:
            return _image_to_string(self._placeholder_image)
        elif isinstance(frame, DetectionResult):
            return {
                field.name: _image_to_string(getattr(frame, field.name))
                for field in fields(frame)  # type: ignore
            }
        return _image_to_string(frame)


def _load_image(image_file: Path) -> Image:
    return cv2.imread(str(image_file), flags=cv2.IMREAD_COLOR_RGB)


def _image_to_string(image: Image) -> str:
    image = convert_rgb_to_bgr(image)
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to convert image to bytes.")
    b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_image}"
