import time
import httpx
from nicegui import ui, app
from pathlib import Path

from camera import Camera
from api import setup_api
from src.demo_interface.image_processing import BasicProcessor, CalibrationProcessor, AnomalyDetectorProcessor


BASE_PATH = Path(__file__).parent
ZAKID_LOGO = BASE_PATH / "assets" / "logo_zakid.png"
UDE_LOGO = BASE_PATH / "assets" / "logo_ude.png"
PLACEHOLDER_IMAGE = BASE_PATH / "assets" / "placeholder.png"
ENCODER_VISUALIZATION = BASE_PATH / "assets" / "encoder_visualization.jpeg"

camera_instance = Camera()
basic_processor = BasicProcessor()
calibration_processor = CalibrationProcessor()
anomaly_detector_processor = AnomalyDetectorProcessor()


def header() -> None:
    """
    Header with logos und title. To use in every subpage.
    """
    with ui.row().classes('w-full items-start'):
        ui.image(ZAKID_LOGO).props('width=200px height=50px fit=scale-down')
        ui.label('Cookie Anomaly Detection').classes('flex-grow text-center self-end text-2xl font-bold')
        ui.image(UDE_LOGO).props('width=200px height=50px fit=scale-down')

    ui.separator()


@ui.page("/")
def index_page() -> None:
    """
    Standard root page. Automatically redirects to the anomaly detection page.
    """
    ui.run_javascript("window.location='/anomaly-detection'")


@ui.page("/basic")
def basis_page() -> None:
    """
    Basic image processing page. Image from the camera is just returned and displayed.
    """
    header()

    result_image = (ui.interactive_image()
                    .classes('w-[800px] h-[800px]')
                    .style('object-fit: contain'))

    async def update_images():
        url = f"http://localhost:8080/video/frame/basic?ts={time.time()}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        data = response.json()
        result_image.set_source(f"data:image/jpeg;base64,{data['result']}")

    ui.timer(interval=0.1, callback=update_images)


@ui.page("/calibration")
def calibration_page() -> None:
    """
    Calibration image processing page. Image from the camera is returned with a red circle, which the .
    """
    header()
    with ui.column().classes('w-full items-center mt-10'):
        result_image = (ui.interactive_image()
                        .classes('w-[800px] h-[800px]')
                        .style('object-fit: contain'))

    async def update_images():
        url = f"http://localhost:8080/video/frame/calibration?ts={time.time()}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        data = response.json()
        result_image.set_source(f"data:image/jpeg;base64,{data['result']}")

    ui.timer(interval=0.1, callback=update_images)


@ui.page("/anomaly-detection")
def anomaly_detection_page() -> None:
    """
    Anomaly Detection page. Detected anomalies and intermediate image processing steps are displayed.
    """
    header()

    with ui.column().classes('w-full items-center mt-10'):

        with ui.row().classes('w-full justify-center'):
            result_image = ui.interactive_image().classes('w-[800px] h-[800px]').style('object-fit: contain')

        with ui.row().classes('w-full items-center justify-center gap-6 mt-7'):
            original_image = ui.interactive_image().classes('w-[150px] h-[150px]').style('object-fit: contain')
            ui.icon("arrow_right").classes("text-4xl font-bold my-auto")
            preprocessed_image = ui.interactive_image().classes('w-[150px] h-[150px]').style('object-fit: contain')
            ui.icon("arrow_right").classes("text-4xl font-bold my-auto")
            ui.image(str(ENCODER_VISUALIZATION)).classes('w-[200px] h-[150px]').style('object-fit: contain')
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


        ui.timer(interval=0.1, callback=update_images)


def on_startup():
    setup_api(
        camera=camera_instance,
        basic_processor=basic_processor,
        calibration_processor=calibration_processor,
        anomaly_detector_processor=anomaly_detector_processor,
        placeholder_image=PLACEHOLDER_IMAGE
    )

app.on_startup(on_startup)
ui.run()
