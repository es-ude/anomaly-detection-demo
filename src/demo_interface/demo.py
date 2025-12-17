import asyncio
import os
from pathlib import Path

from nicegui import app, ui

from src.demo_interface.demo_application_controller import DemoApplicationController
from src.demo_interface.image_processing import (
    AnomalyDetectorProcessor,
    BasicProcessor,
    CalibrationProcessor,
)

ASSETS_DIR = Path(__file__).parent / "assets"
ZAKID_LOGO = ASSETS_DIR / "zakid_logo_weiß.svg"
UDE_LOGO = ASSETS_DIR / "logo_ude_weiß_transparent.svg"
PLACEHOLDER_IMAGE = ASSETS_DIR / "placeholder_2.png"
ENCODER_VISUALIZATION = ASSETS_DIR / "darstellung_encoder_decoder_weiß.png"

HEIGHT_BIG_IMAGE = 700
WIDTH_BIG_IMAGE = 700
HEIGHT_SMALL_IMAGE_CONTAINER = 200
WIDTH_SMALL_IMAGE_CONTAINER = 200
SMALL_IMAGE_LENGTH = 175

USE_PICAM_MODULE = "ENABLE_PI_CAM" in os.environ
CAM_PORT = int(os.environ.get("CAM_PORT", 0))
IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])

AE_MODEL_CKPT = Path(os.environ["COOKIE_CKPT_DIR"]) / "ae_model.pt"
CLF_MODEL_CKPT = Path(os.environ["COOKIE_CKPT_DIR"]) / "clf_model.pt"
USE_CLASSIFIER = "USE_CLASSIFIER" in os.environ


def setup() -> None:
    app_controller = DemoApplicationController(
        cam_port=CAM_PORT,
        placeholder_image_file=PLACEHOLDER_IMAGE,
        use_picam=USE_PICAM_MODULE,
    )
    basic_processor = BasicProcessor(target_image_size=(800, 800))
    calibration_processor = CalibrationProcessor(target_image_size=(800, 800))
    anomaly_detector_processor = AnomalyDetectorProcessor(
        autoencoder_file=AE_MODEL_CKPT,
        classifier_file=CLF_MODEL_CKPT if USE_CLASSIFIER else None,
        target_image_size=(800, 800),
        inference_image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
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

        with ui.column().classes("w-full items-center"):
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

        with ui.column().classes("w-full items-center"):
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

        with ui.column().classes("w-full items-center"):
            with ui.row().classes(
                "w-full justify-center gap-6" + ("" if USE_CLASSIFIER else " collapse")
            ):
                with ui.element("div").classes(
                    "bg-blue-700 text-blue-100 rounded-full text-xl font-medium px-2.5 py-0.5"
                ) as condition_frame:
                    condition_text = ui.label()

            with ui.row().classes("w-full justify-center gap-6"):
                with ui.column().classes("items-center justify-center gap-6"):
                    residuals_image = ui.interactive_image().classes(
                        f"w-[{WIDTH_BIG_IMAGE}]px] h-[{HEIGHT_BIG_IMAGE}px]"
                    )
                with ui.column().classes("items-center justify-center gap-6"):
                    result_image = ui.interactive_image().classes(
                        f"w-[{WIDTH_BIG_IMAGE}]px] h-[{HEIGHT_BIG_IMAGE}px]"
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

        def update_images(result: dict[str, str | bool | None] | str) -> None:
            def get_image(key: str) -> str:
                if isinstance(result, str):
                    return result
                value = result[key]
                if not isinstance(value, str):
                    raise ValueError("Not an base64 encoded image.")
                return value

            def display_condition() -> None:
                frame_template = "bg-{0}-700 text-{0}-100 rounded-full text-xl font-medium px-2.5 py-0.5"
                if isinstance(result, str) or result["damaged"] is None:
                    condition_frame.classes(replace=frame_template.format("blue"))
                    condition_text.set_text("Unbekannt")
                elif result["damaged"]:
                    condition_frame.classes(replace=frame_template.format("red"))
                    condition_text.set_text("Beschädigt")
                else:
                    condition_frame.classes(replace=frame_template.format("green"))
                    condition_text.set_text("Unbeschädigt")

            result_image.set_source(get_image("superimposed"))
            original_image.set_source(get_image("original"))
            preprocessed_image.set_source(get_image("preprocessed"))
            reconstructed_image.set_source(get_image("reconstructed"))
            residuals_image.set_source(get_image("residuals"))
            mini_residuals_image.set_source(get_image("residuals"))
            result_mini_image.set_source(get_image("superimposed"))
            display_condition()

        await app_controller.set_handler(anomaly_detector_processor, update_images)

    asyncio.create_task(app_controller.run())
    app.on_shutdown(app_controller.close)


app.on_startup(setup)
ui.run(title="CookieAdDemo", favicon="🍪")
