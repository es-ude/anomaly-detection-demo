import asyncio
import base64
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
from nicegui import run

from demo.anomaly_detection.anomaly_detector import DetectionResult
from demo.camera.image import Image, convert_rgb_to_bgr
from demo.web_app.image_processing import ImageProcessor

type UpdateUICallback = Callable[[dict[str, str]], None] | Callable[[str], None]


class _NoneImageProcessor:
    def process(self, image: Image | None) -> None:
        return None


class DemoApplicationController:
    def __init__(
        self, cam_port: int | str, placeholder_image_file: Path, use_picam: bool = False
    ) -> None:
        from demo.camera.camera import Camera as ICamera

        if use_picam:
            from demo.camera.picamv3_camera import PiCamera as Camera
        else:
            from demo.camera.opencv_camera import CVCamera as Camera

        self._camera: ICamera = Camera(cam_port, width=1920, height=1080)
        self._placeholder_image = _load_image(placeholder_image_file)
        self._image_processor: ImageProcessor = _NoneImageProcessor()
        self._update_ui_callback: UpdateUICallback = lambda _: None
        self._main_lock = asyncio.Lock()

    async def run(self) -> None:
        while True:
            async with self._main_lock:
                processed_frame = await self._take_and_process_frame()
                ui_data = self._frame_to_ui_data(processed_frame)
                self._update_ui_callback(ui_data)  # type: ignore

    async def set_handler(
        self, image_processor: ImageProcessor, ui_callback: UpdateUICallback
    ) -> None:
        async with self._main_lock:
            self._image_processor = image_processor
            self._update_ui_callback = ui_callback

    def close(self) -> None:
        self._camera.release()

    async def _take_and_process_frame(self) -> DetectionResult | Image | None:
        frame = await run.io_bound(self._camera.read_frame)
        return await run.cpu_bound(self._image_processor.process, frame)

    def _frame_to_ui_data(
        self, frame: DetectionResult | Image | None
    ) -> dict[str, str | bool | None] | str:
        if frame is None:
            return _image_to_string(self._placeholder_image)
        elif isinstance(frame, DetectionResult):
            return _detection_result_to_ui_data(frame)
        return _image_to_string(frame)


def _load_image(image_file: Path) -> Image:
    img = cv2.imread(str(image_file), flags=cv2.IMREAD_COLOR_RGB)
    if img is None:
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return img


def _image_to_string(image: Image) -> str:
    image = convert_rgb_to_bgr(image)
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to convert image to bytes.")
    b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_image}"


def _detection_result_to_ui_data(
    detection_result: DetectionResult,
) -> dict[str, str | bool | None]:
    ui_data = dict()
    for field in fields(detection_result):
        value = getattr(detection_result, field.name)
        ui_data[field.name] = (
            value
            if value is None or isinstance(value, bool)
            else _image_to_string(value)
        )
    return ui_data
