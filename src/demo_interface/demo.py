import asyncio
from os import environ
from pathlib import Path

from nicegui import app, ui

from src.demo_interface.demo_application_controller import DemoApplicationController
from src.demo_interface.image_processing import (
    AnomalyDetectorProcessor,
    BasicProcessor,
    CalibrationProcessor,
)

BASE_PATH = Path(__file__).parent
ZAKID_LOGO = BASE_PATH / "assets" / "zakid_logo_weiß.svg"
UDE_LOGO = BASE_PATH / "assets" / "logo_ude_weiß_transparent.svg"
PLACEHOLDER_IMAGE = BASE_PATH / "assets" / "placeholder_2.png"
ENCODER_VISUALIZATION = BASE_PATH / "assets" / "darstellung_encoder_decoder_weiß.png"

USE_RASPBERRY_CAMERA_MODULE = environ.get("ENABLE_PI_CAM", False)
HEIGHT_BIG_IMAGE = 750
WIDTH_BIG_IMAGE = 750
HEIGHT_SMALL_IMAGE_CONTAINER = 200
WIDTH_SMALL_IMAGE_CONTAINER = 200
SMALL_IMAGE_LENGTH = 175


def setup() -> None:
    app_controller = DemoApplicationController(
        cam_port=int(environ.get("CAM_PORT", 0)),
        placeholder_image_file=PLACEHOLDER_IMAGE,
        use_picam=USE_RASPBERRY_CAMERA_MODULE,
    )
    basic_processor = BasicProcessor(target_image_size=(800, 800))
    calibration_processor = CalibrationProcessor(target_image_size=(800, 800))
    anomaly_detector_processor = AnomalyDetectorProcessor(
        target_image_size=(800, 800),
        inference_image_size=(128, 128),
        model_file=Path(__file__).parents[2]
        / "src/anomaly_detection/model_checkpoints/cookie/model.pt",
    )

    def _header() -> None:
        """
        Header with logos und title. To use in every subpage. Also adds background color.
        """
        ui.add_head_html("<style>body {background-color: #222; }</style>")
        with ui.row().classes("w-full items-start"):
            ui.image(ZAKID_LOGO).props("width=200px height=50px fit=scale-down")
            ui.label("KI-basierte Defektkontrolle").classes(
                "flex-grow text-center self-end text-4xl font-bold"
            ).style("color: white;")

            ui.image(UDE_LOGO).props("width=200px height=50px fit=scale-down")

        ui.separator()

    @ui.page("/")
    def index_page() -> None:
        """
        Standard root page. Automatically redirects to the anomaly detection page.
        """
        ui.run_javascript("window.location='/anomaly-detection'")

    @ui.page("/basic")
    async def basic_page() -> None:
        """
        Basic image processing page. Image from the camera is just returned and displayed.
        """
        _header()

        result_image = (
            ui.interactive_image()
            .classes(f"w-[{WIDTH_BIG_IMAGE}]px] h-[{HEIGHT_BIG_IMAGE}px]")
            .style("object-fit: contain")
        )

        def update_images(result: str) -> None:
            result_image.set_source(result)

        await app_controller.set_handler(basic_processor, update_images)

    @ui.page("/calibration")
    async def calibration_page() -> None:
        """
        Calibration image processing page. Image from the camera is returned with a red circle, which the .
        """
        _header()

        with ui.column().classes("w-full items-center mt-10"):
            result_image = (
                ui.interactive_image()
                .classes(f"w-[{WIDTH_BIG_IMAGE}]px] h-[{HEIGHT_BIG_IMAGE}px]")
                .style("object-fit: contain")
            )

        def update_images(result: str) -> None:
            result_image.set_source(result)

        await app_controller.set_handler(calibration_processor, update_images)

    @ui.page("/anomaly-detection")
    async def anomaly_detection_page() -> None:
        """
        Anomaly Detection page. Detected anomalies and intermediate image processing steps are displayed.
        """
        _header()

        with ui.column().classes("w-full items-center mt-1"):
            with ui.row().classes("w-full justify-center gap-20"):
                residuals_image = (
                    ui.interactive_image()
                    .classes(f"w-[{WIDTH_BIG_IMAGE}]px] h-[{HEIGHT_BIG_IMAGE}px]")
                    .style("object-fit: contain")
                )
                result_image = (
                    ui.interactive_image()
                    .classes(f"w-[{WIDTH_BIG_IMAGE}]px] h-[{HEIGHT_BIG_IMAGE}px]")
                    .style("object-fit: contain")
                )

            with ui.row().classes("w-full items-center justify-center gap-6 mt-1"):
                with ui.column().classes(
                    f"w-[{WIDTH_SMALL_IMAGE_CONTAINER}px] h-[{HEIGHT_SMALL_IMAGE_CONTAINER}px] items-center justify-center"
                ):
                    original_image = (
                        ui.interactive_image()
                        .classes(
                            f"w-[{SMALL_IMAGE_LENGTH}px] h-[{SMALL_IMAGE_LENGTH}px]"
                        )
                        .style("object-fit: contain")
                    )
                    ui.label("Original").style(
                        "text-align: center; width: 100%; font-weight: bold; color: white;"
                    )

                ui.icon("arrow_right").classes("text-4xl font-bold my-auto").style(
                    "color: white; margin-top: 65px;"
                )

                with ui.column().classes(
                    f"w-[{WIDTH_SMALL_IMAGE_CONTAINER}px] h-[{HEIGHT_SMALL_IMAGE_CONTAINER}px] items-center justify-center"
                ):
                    preprocessed_image = (
                        ui.interactive_image()
                        .classes(
                            f"w-[{SMALL_IMAGE_LENGTH}px] h-[{SMALL_IMAGE_LENGTH}px]"
                        )
                        .style("object-fit: contain")
                    )
                    ui.label("Vorverarbeitet").style(
                        "text-align: center; width: 100%; font-weight: bold; color: white;"
                    )

                ui.icon("arrow_right").classes("text-4xl font-bold my-auto").style(
                    "color: white; margin-top: 65px;"
                )

                with ui.column().classes(
                    f"w-[{WIDTH_SMALL_IMAGE_CONTAINER}px] h-[{HEIGHT_SMALL_IMAGE_CONTAINER}px] items-center justify-center"
                ):
                    ui.image(str(ENCODER_VISUALIZATION)).classes(
                        f"w-[{SMALL_IMAGE_LENGTH}px] h-[{SMALL_IMAGE_LENGTH}px]"
                    ).props("fit=scale-down")
                    ui.label("Convolutional Autoencoder").style(
                        "text-align: center; width: 100%; font-weight: bold; color: white;"
                    )

                ui.icon("arrow_right").classes("text-4xl font-bold my-auto").style(
                    "color: white; margin-top: 65px;"
                )

                with ui.column().classes(
                    f"w-[{WIDTH_SMALL_IMAGE_CONTAINER}px] h-[{HEIGHT_SMALL_IMAGE_CONTAINER}px] items-center justify-center"
                ):
                    reconstructed_image = (
                        ui.interactive_image()
                        .classes(
                            f"w-[{SMALL_IMAGE_LENGTH}px] h-[{SMALL_IMAGE_LENGTH}px]"
                        )
                        .style("object-fit: contain")
                    )
                    ui.label("Rekonstruiert").style(
                        "text-align: center; width: 100%; font-weight: bold; color: white;"
                    )

                ui.icon("arrow_right").classes("text-4xl font-bold my-auto").style(
                    "color: white; margin-top: 65px;"
                )

                with ui.column().classes(
                    f"w-[{WIDTH_SMALL_IMAGE_CONTAINER}px] h-[{HEIGHT_SMALL_IMAGE_CONTAINER}px] items-center justify-center"
                ):
                    mini_residuals_image = (
                        ui.interactive_image()
                        .classes(
                            f"w-[{SMALL_IMAGE_LENGTH}px] h-[{SMALL_IMAGE_LENGTH}px]"
                        )
                        .style("object-fit: contain")
                    )
                    ui.label("|Vorverarbeitet - Rekonstruiert|").style(
                        "text-align: center; width: 100%; font-weight: bold; color: white;"
                    )

                ui.icon("arrow_right").classes("text-4xl font-bold my-auto").style(
                    "color: white; margin-top: 65px;"
                )

                with ui.column().classes(
                    f"w-[{WIDTH_SMALL_IMAGE_CONTAINER}px] h-[{HEIGHT_SMALL_IMAGE_CONTAINER}px] items-center justify-center"
                ):
                    result_mini_image = (
                        ui.interactive_image()
                        .classes(
                            f"w-[{SMALL_IMAGE_LENGTH}px] h-[{SMALL_IMAGE_LENGTH}px]"
                        )
                        .style("object-fit: contain")
                    )
                    ui.label("Ergebnis mit Anomalien").style(
                        "text-align: center; width: 100%; font-weight: bold; color: white;"
                    )

        def update_images(result: dict[str, str] | str) -> None:
            def get_image(key: str) -> str:
                return result if isinstance(result, str) else result[key]

            result_image.set_source(get_image("superimposed"))
            original_image.set_source(get_image("original"))
            preprocessed_image.set_source(get_image("preprocessed"))
            reconstructed_image.set_source(get_image("reconstructed"))
            residuals_image.set_source(get_image("residuals"))
            mini_residuals_image.set_source(get_image("residuals"))
            result_mini_image.set_source(get_image("superimposed"))

        await app_controller.set_handler(anomaly_detector_processor, update_images)

    asyncio.create_task(app_controller.run())
    app.on_shutdown(app_controller.close)


app.on_startup(setup)
ui.run(reload=True, show=False)
