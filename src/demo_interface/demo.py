import time

from nicegui import ui, app
from pathlib import Path
from nicegui.events import ValueChangeEventArguments

from camera import Camera
from api import setup_api
from image_processing import CookieCalibrator, ImageBaseProcessor

zakid_logo = Path(__file__).parent / "assets" / "logo_zakid.png"
ude_logo = Path(__file__).parent / "assets" / "logo_ude.png"
placeholder_image = Path(__file__).parent / "assets" / "placeholder.png"

camera_instance = Camera()
image_processor = ImageBaseProcessor()
# image_processor = CookieCalibrator()


def setup() -> None:

    with ui.row().classes('w-full items-start'):
        ui.image(zakid_logo).props('width=200px height=50px fit=scale-down')
        ui.label('Cookie Anomaly Detection').classes('flex-grow text-center self-end text-2xl font-bold')
        ui.image(ude_logo).props('width=200px height=50px fit=scale-down')

    ui.separator()

    setup_api(camera=camera_instance, image_processor=image_processor, placeholder_image=placeholder_image)
    with ui.column().classes('w-full items-center mt-4'):
        video_image = ui.interactive_image().classes('w-[800px] h-[800px]').style('object-fit: contain')
        ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))

app.on_startup(setup)
ui.run()
