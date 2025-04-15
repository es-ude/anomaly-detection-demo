import asyncio
from typing import Callable, Dict

from nicegui import run
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor

from src.demo_interface.camera import Camera
from src.demo_interface.image_processing import AbstractImageProcessor
from src.demo_interface.utils import convert_bytes_to_base64

cpu_executor = ProcessPoolExecutor(max_workers=10)

class FrameData:

    def __init__(self, processor: AbstractImageProcessor):
        self._result = None
        self._processor = processor
        self._callback = None

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        if self._callback is not None:
            self._callback(result)
        self._result = result

    @property
    def processor(self):
        return self._processor

    @processor.setter
    def processor(self, new_frame_processor: AbstractImageProcessor):
        self._processor = new_frame_processor

    def set_callback(self, callback: Callable[[Dict], None]):
        self._callback = callback

def _to_dict_response(processed, placeholder_bytes):
    if processed is None:
        return {"result": convert_bytes_to_base64(placeholder_bytes)}

    data_dict = {
        key: convert_bytes_to_base64(img_bytes) \
            if img_bytes is not None else convert_bytes_to_base64(placeholder_bytes) \
        for key, img_bytes in asdict(processed).items()
    }
    return data_dict


async def cpu_bound(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(cpu_executor, func, *args, **kwargs)


async def _handle_frame_request(
    camera: Camera,
    processor: AbstractImageProcessor,
    placeholder_bytes: bytes,
) -> dict:

    if not camera.is_opened():
        return {"result": convert_bytes_to_base64(placeholder_bytes)}

    frame = await run.io_bound(camera.read_frame)
    if frame is None:
        return {"result": convert_bytes_to_base64(placeholder_bytes)}

    processed = await cpu_bound(processor.process_frame, frame)

    return _to_dict_response(processed, placeholder_bytes)


async def start_frame_updates(
        frame_data: FrameData,
        camera: Camera,
        placeholder_bytes: bytes,
) -> None:
    while True:
        result = await _handle_frame_request(camera, frame_data.processor, placeholder_bytes)
        frame_data.result = result
        await asyncio.sleep(0.1)