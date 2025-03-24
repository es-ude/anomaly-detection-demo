import time
import httpx
from nicegui import ui, app
from pathlib import Path

from camera import Camera
from api import setup_api
from src.demo_interface.image_processing import BaseProcessor, CalibrationProcessor, AnomalyDetectorProcessor

base_path = Path(__file__).parent
zakid_logo = base_path / "assets" / "logo_zakid.png"
ude_logo = base_path / "assets" / "logo_ude.png"
placeholder_image = base_path / "assets" / "placeholder.png"
encoder_visualization = base_path / "assets" / "encoder_visualization.jpeg"

camera_instance = Camera()
base_processor = BaseProcessor()
calibration_processor = CalibrationProcessor()
anomaly_detector_processor = AnomalyDetectorProcessor()


def setup() -> None:

    with ui.row().classes('w-full items-start'):
        ui.image(zakid_logo).props('width=200px height=50px fit=scale-down')
        ui.label('Cookie Anomaly Detection').classes('flex-grow text-center self-end text-2xl font-bold')
        ui.image(ude_logo).props('width=200px height=50px fit=scale-down')

    ui.separator()

    setup_api(
        camera=camera_instance,
        base_processor=base_processor,
        calibration_processor=calibration_processor,
        anomaly_detector_processor=anomaly_detector_processor,
        placeholder_image=placeholder_image
    )

    with ui.column().classes('w-full items-center mt-10'):
        with ui.row().classes('w-full justify-center'):
            result_image = ui.interactive_image().classes('w-[800px] h-[800px]').style('object-fit: contain')
        with ui.row().classes('w-full items-center justify-center gap-6 mt-5'):
            original_image = ui.interactive_image().classes('w-[150px] h-[150px]').style('object-fit: contain')
            ui.icon("arrow_right").classes("text-4xl font-bold my-auto")
            preprocessed_image = ui.interactive_image().classes('w-[150px] h-[150px]').style('object-fit: contain')
            ui.icon("arrow_right").classes("text-4xl font-bold my-auto")
            ui.image(str(encoder_visualization)).classes('w-[200px] h-[150px]').style('object-fit: contain')
            ui.icon("arrow_right").classes("text-4xl font-bold my-auto")
            reconstructed_image = ui.interactive_image().classes('w-[150px] h-[150px]').style('object-fit: contain')
            ui.icon("arrow_right").classes("text-4xl font-bold my-auto")
            residuals_image = ui.interactive_image().classes('w-[150px] h-[150px]').style('object-fit: contain')
            ui.icon("arrow_right").classes("text-4xl font-bold my-auto")
            result_mini_image = ui.interactive_image().classes('w-[150px] h-[150px]').style('object-fit: contain')


        async def update_images():
            now = time.time()
            url = f"http://localhost:8080/video/frame/anomaly-detection?ts={now}"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)

            data = response.json()
            result_image.set_source(f"data:image/jpeg;base64,{data['result']}")
            original_image.set_source(f"data:image/jpeg;base64,{data['original']}")
            preprocessed_image.set_source(f"data:image/jpeg;base64,{data['preprocessed']}")
            reconstructed_image.set_source(f"data:image/jpeg;base64,{data['reconstructed']}")
            residuals_image.set_source(f"data:image/jpeg;base64,{data['residuals']}")
            result_mini_image.set_source(f"data:image/jpeg;base64,{data['result']}")


        # ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))
        ui.timer(interval=0.1, callback=update_images)

app.on_startup(setup)
ui.run()
