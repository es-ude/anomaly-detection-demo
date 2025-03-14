import signal

from fastapi import Response
from nicegui import Client, app, core, ui, run
from pathlib import Path

from src.demo_interface.camera import Camera
from src.demo_interface.image_processing import ImageBaseProcessor


def _load_image_as_bytes(image_path: Path) -> bytes:
    with open(image_path, "rb") as file:
        return file.read()


def setup_api(camera:  Camera, image_processor: ImageBaseProcessor, placeholder_image: Path):
    placeholder = Response(content=_load_image_as_bytes(placeholder_image), media_type='image/png')

    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not camera.is_opened():
            return placeholder

        frame = await run.io_bound(camera.read_frame)
        if frame is None:
            return placeholder

        jpeg = await run.cpu_bound(image_processor.process_frame, frame)
        if jpeg is None:
            return placeholder
        return Response(content=jpeg, media_type='image/jpeg')

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
