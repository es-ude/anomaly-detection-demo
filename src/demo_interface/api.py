import signal
import base64
import time

import numpy as np
import cv2

from fastapi import Response
from starlette.responses import JSONResponse
from nicegui import Client, app, core, ui, run
from pathlib import Path
from dataclasses import asdict

from src.anomaly_detector import DetectionResult
from src.demo_interface.camera import Camera
from src.demo_interface.image_processing import ImageBaseProcessor
from src.demo_interface.utils import load_image_as_bytes, convert_bytes_to_base64


def setup_api(camera:  Camera, image_processor: ImageBaseProcessor, placeholder_image: Path):
    placeholder_bytes = load_image_as_bytes(placeholder_image)
    placeholder_json_response = JSONResponse({"result": convert_bytes_to_base64(placeholder_bytes),})

    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not camera.is_opened():
            return placeholder_json_response

        frame = await run.io_bound(camera.read_frame)
        if frame is None:
            return placeholder_json_response

        image_result = await run.cpu_bound(image_processor.process_frame, frame)

        if image_result is None:
            return JSONResponse({"result": convert_bytes_to_base64(load_image_as_bytes(placeholder_image)),})

        image_result = {
            key: convert_bytes_to_base64(img_bytes) \
            if img_bytes is not None else convert_bytes_to_base64(placeholder_bytes) \
            for key, img_bytes in asdict(image_result).items()
        }
        return JSONResponse(image_result)

    async def disconnect() -> None:
        for client_id in Client.instances:
            await core.sio.disconnect(client_id)

    def handle_sigint(signum, frame) -> None:
        ui.timer(0.1, disconnect, once=True)
        ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

    async def cleanup() -> None:
        await disconnect()
        camera.release()

    app.on_shutdown(cleanup)
    signal.signal(signal.SIGINT, handle_sigint)
