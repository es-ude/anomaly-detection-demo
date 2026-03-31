import os
from pathlib import Path

from nicegui import ui

from demo.web_app.controller.image_processing import AnomalyDetectorProcessor
from demo.web_app.pages.layout import page_layout

AE_MODEL_CKPT = Path(os.getenv("COOKIE_CKPT_DIR", "")) / "ae_model.pt"
USE_CLASSIFIER = os.getenv("USE_CLASSIFIER", "").lower() in ["true", "1"]
DEVICE = os.getenv("DEVICE", "cpu")
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", 1080))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", 1920))
FLIP_HORIZONTAL = os.getenv("FLIP_HORIZONTAL", "").lower() in ["true", "1"]
FLIP_VERTICAL = os.getenv("FLIP_VERTICAL", "").lower() in ["true", "1"]

ASSETS_DIR = Path(__file__).parent / "assets"
ENCODER_VISUALIZATION = ASSETS_DIR / "darstellung_encoder_decoder_weiß.png"

DIMENSION_LARGE_IMAGE = "w-[60vh] h-[60vh]"
DIMENSION_SMALL_IMAGE = "w-[10vh] h-[10vh]"

anomaly_detector_processor = AnomalyDetectorProcessor(
    autoencoder_file=AE_MODEL_CKPT,
    use_classifier=USE_CLASSIFIER,
    target_image_size=(800, 800),
    inference_image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    device=DEVICE,
    flip_horizontal=FLIP_HORIZONTAL,
    flip_vertical=FLIP_VERTICAL,
)


def get_anomaly_detection_processor() -> AnomalyDetectorProcessor:
    return anomaly_detector_processor


async def anomaly_detection(app_controller):
    page_layout()

    with ui.row().classes(
        "w-full justify-center py-4" + ("" if USE_CLASSIFIER else " collapse")
    ):
        with ui.element("div").classes(
            "bg-blue-700 text-blue-100 rounded-full text-xl font-medium px-2.5 py-0.5"
        ) as condition_frame:
            condition_text = ui.label()

    with ui.row(align_items="center").classes("w-full justify-center-safe py-4 px-2"):
        with ui.column():
            residuals_image = ui.interactive_image().classes(DIMENSION_LARGE_IMAGE)
        with ui.column():
            result_image = ui.interactive_image().classes(DIMENSION_LARGE_IMAGE)

    with ui.row(align_items="center").classes("w-full justify-center-safe px-2 py-4"):
        with ui.column(align_items="center"):
            original_image = (
                ui.interactive_image()
                .classes(DIMENSION_SMALL_IMAGE)
                .style("object-fit: contain")
            )
            ui.label("Original").classes("text-white font-bold")

        with ui.column():
            ui.icon("arrow_right").classes("text-4xl font-bold text-white")

        with ui.column(align_items="center"):
            preprocessed_image = (
                ui.interactive_image()
                .classes(DIMENSION_SMALL_IMAGE)
                .style("object-fit: contain")
            )
            ui.label("Vorverarbeitet").classes("text-bold text-white")

        with ui.column():
            ui.icon("arrow_right").classes("text-4xl font-bold text-white")

        with ui.column().classes("items-center justify-center"):
            ui.image(str(ENCODER_VISUALIZATION)).classes(DIMENSION_SMALL_IMAGE).props(
                "fit=scale-down"
            )
            ui.label("Convolutional Autoencoder").classes("text-bold text-white")

        with ui.column():
            ui.icon("arrow_right").classes("text-4xl font-bold text-white")

        with ui.column().classes("items-center justify-center"):
            reconstructed_image = (
                ui.interactive_image()
                .classes(DIMENSION_SMALL_IMAGE)
                .style("object-fit: contain")
            )
            ui.label("Rekonstruiert").classes("text-bold text-white")

        with ui.column():
            ui.icon("arrow_right").classes("text-4xl font-bold text-white")

        with ui.column().classes("items-center justify-center"):
            mini_residuals_image = (
                ui.interactive_image()
                .classes(DIMENSION_SMALL_IMAGE)
                .style("object-fit: contain")
            )
            ui.label("| Vorverarbeitet - Rekonstruiert |").classes(
                "text-bold text-white"
            )

        with ui.column():
            ui.icon("arrow_right").classes("text-4xl font-bold text-white")

        with ui.column().classes("items-center justify-center"):
            result_mini_image = (
                ui.interactive_image()
                .classes(DIMENSION_SMALL_IMAGE)
                .style("object-fit: contain")
            )
            ui.label("Ergebnis mit Anomalien").classes("text-bold text-white")

    def update_images(result: dict[str, str | bool | None] | str) -> None:
        def get_image(key: str) -> str:
            if isinstance(result, str):
                return result
            value = result[key]
            if not isinstance(value, str):
                raise ValueError("Not an base64 encoded image.")
            return value

        def display_condition() -> None:
            frame_template = (
                "bg-{0}-700 text-white rounded-full text-xl font-medium px-2.5 py-0.5"
            )
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
