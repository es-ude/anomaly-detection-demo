import time
import httpx
from nicegui import ui, app
from pathlib import Path

from camera import Camera
from api import setup_api
from src.demo_interface.image_processing import BasicProcessor, CalibrationProcessor, AnomalyDetectorProcessor


BASE_PATH = Path(__file__).parent
ZAKID_LOGO = BASE_PATH / "assets" / "zakid_logo_weiß.svg"
UDE_LOGO = BASE_PATH / "assets" / "logo_ude_weiß_transparent.svg"
PLACEHOLDER_IMAGE = BASE_PATH / "assets" / "placeholder.png"
ENCODER_VISUALIZATION = BASE_PATH / "assets" / "darstellung_encoder_decoder_weiß.png"

camera_instance = Camera()
basic_processor = BasicProcessor()
calibration_processor = CalibrationProcessor()
anomaly_detector_processor = AnomalyDetectorProcessor()


def header() -> None:
    """
    Header with logos und title. To use in every subpage. Also adds background color.
    """
    ui.add_head_html('<style>body {background-color: #222; }</style>')
    with ((ui.row().classes('w-full items-start'))):
        ui.image(ZAKID_LOGO).props('width=200px height=50px fit=scale-down')
        ui.label('KI-basierte Defektkontrolle').classes('flex-grow text-center self-end text-4xl font-bold').style("color: white;")

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

        with ui.row().classes('w-full justify-center gap-5'):
            residuals_image = ui.interactive_image().classes('w-[800px] h-[800px]').style('object-fit: contain')
            result_image = ui.interactive_image().classes('w-[800px] h-[800px]').style('object-fit: contain')

        with ui.row().classes('w-full items-center justify-center gap-6 mt-7'):
            with ui.column().classes('w-[200px] h-[200px] items-center justify-center'):
                original_image = ui.interactive_image() \
                    .classes('w-full h-[150px]') \
                    .style('object-fit: contain')
                ui.label("Original") \
                    .style("text-align: center; width: 100%; font-weight: bold; color: white;")

            ui.icon("arrow_right") \
                .classes("text-4xl font-bold my-auto") \
                .style("color: white; margin-top: 65px;")

            with ui.column().classes('w-[200px] h-[200px] items-center justify-center'):
                preprocessed_image = ui.interactive_image() \
                    .classes('w-full h-[150px]') \
                    .style('object-fit: contain')
                ui.label("Vorverarbeitet") \
                    .style("text-align: center; width: 100%; font-weight: bold; color: white;")

            ui.icon("arrow_right") \
                .classes("text-4xl font-bold my-auto") \
                .style("color: white; margin-top: 65px;")

            with ui.column().classes('w-[200px] h-[200px] items-center justify-center'):
                ui.image(str(ENCODER_VISUALIZATION)) \
                    .classes('w-full h-[150px]') \
                    .props("fit=scale-down")
                ui.label("Convolutional Autoencoder") \
                    .style("text-align: center; width: 100%; font-weight: bold; color: white;")

            ui.icon("arrow_right") \
                .classes("text-4xl font-bold my-auto") \
                .style("color: white; margin-top: 65px;")

            with ui.column().classes('w-[200px] h-[200px] items-center justify-center'):
                reconstructed_image = ui.interactive_image() \
                    .classes('w-full h-[150px]') \
                    .style('object-fit: contain')
                ui.label("Rekonstruiert") \
                    .style("text-align: center; width: 100%; font-weight: bold; color: white;")

            ui.icon("arrow_right") \
                .classes("text-4xl font-bold my-auto") \
                .style("color: white; margin-top: 65px;")

            with ui.column().classes('w-[200px] h-[200px] items-center justify-center'):
                mini_residuals_image = ui.interactive_image() \
                    .classes('w-full h-[150px]') \
                    .style('object-fit: contain')
                ui.label("|Vorverarbeitet - Rekonstruiert|") \
                    .style("text-align: center; width: 100%; font-weight: bold; color: white;")

            ui.icon("arrow_right") \
                .classes("text-4xl font-bold my-auto") \
                .style("color: white; margin-top: 65px;")

            with ui.column().classes('w-[200px] h-[200px] items-center justify-center'):
                result_mini_image = ui.interactive_image() \
                    .classes('w-full h-[150px]') \
                    .style('object-fit: contain')
                ui.label("Ergebnis mit Anomalien") \
                    .style("text-align: center; width: 100%; font-weight: bold; color: white;")


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
            mini_residuals_image.set_source(f"data:image/jpeg;base64,{data['residuals']}")
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
    time.sleep(2)

app.on_startup(on_startup)
ui.run()
