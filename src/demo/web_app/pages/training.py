import base64
import uuid
from asyncio import to_thread
from asyncio.locks import Lock
from datetime import datetime
from pathlib import Path

from nicegui import ui

from demo.web_app.controller.demo_application_controller import (
    DemoApplicationController,
)
from demo.web_app.controller.image_processing import (
    AnomalyDetectorProcessor,
    BasicProcessor,
)
from demo.web_app.controller.online_training import train
from demo.web_app.pages.layout import page_layout

image_processor = BasicProcessor(
    target_image_size=(1080, 1920), flip_horizontal=False, flip_vertical=False
)


async def training(
    app_controller: DemoApplicationController,
    anomaly_detection_processor: AnomalyDetectorProcessor,
) -> None:
    page_layout()

    image_src: dict[str, str] = {"source": ""}
    training_in_progess: Lock = Lock()
    tmp_data: Path = Path(__file__).parent.parent.joinpath(
        "recordings", "train", "good"
    )

    async def add_file() -> None:
        if training_in_progess.locked():
            ui.notify(
                "Training in progress. Please wait until it is finished.", color="red"
            )
        else:
            file_path = tmp_data.joinpath(
                f"{datetime.now().isoformat()}_{uuid.uuid1()}.jpg"
            )
            img = base64.b64decode((image_src["source"].split(",")[1]).encode("utf-8"))
            file_path.write_bytes(img)
            files.add_row({"path": file_path.name})
            files.run_method("scrollTo", len(files.rows) - 1)

    async def reset_files() -> None:
        if training_in_progess.locked():
            ui.notify(
                "Training in progress. Please wait until it is finished.", color="red"
            )
        else:
            for file in tmp_data.iterdir():
                file.unlink()
            tmp_data.mkdir(parents=True, exist_ok=True)

            files.rows = []
            files.update()

    async def train_model() -> None:
        if training_in_progess.locked():
            ui.notify(
                "Training already in progress. Please wait until it is finished.",
                color="red",
            )
        else:
            async with training_in_progess:
                ui.notify(
                    f"TRAINING STARTED: {datetime.now().time()}",
                    type="info",
                    close_button="X",
                    timeout=0,
                )
                # TEST: method train actually working
                await to_thread(
                    train,
                    anomaly_detector=anomaly_detection_processor.get_anomaly_detector(),
                    dataset_dir=tmp_data.parent.parent,
                    device="cpu",
                )
                ui.notify(
                    f"TRAINING FINISHED: {datetime.now().time()}",
                    type="positive",
                    close_button="X",
                    timeout=0,
                )

    async def hard_reset() -> None:
        if training_in_progess.locked():
            ui.notify(
                "Training in progress. Please wait until it is finished.", color="red"
            )
        else:
            with ui.dialog() as dialog, ui.card():
                ui.label("Do you really want to reset the anomaly detector?").classes(
                    "text-lg"
                )
                with ui.row().classes("w-full justify-between"):
                    ui.button("Cancel", on_click=dialog.close).classes(
                        "bg-gray-500"
                    ).props("fab")
                    ui.button("OK", on_click=lambda: dialog.submit("OK")).classes(
                        "bg-red"
                    ).props("fab")
            result = await dialog
            if result == "OK":
                anomaly_detection_processor.reset_anomaly_detector()
                ui.notify("ANOMALY DETECTOR RESET DONE")

    with ui.row(align_items="start").classes(
        "w-full h-full justify-center content-start"
    ):
        with ui.column(align_items="center").classes("w-[40vw] h-full"):
            ui.image().classes("h-[60vh]").style(
                "object-fit: contain;"
            ).bind_source_from(image_src).props("no-transition")
            with (
                ui.button(on_click=add_file)
                .classes("w-max mt-4")
                .props("fab")
                .tooltip("take a picture")
            ):
                ui.icon("camera_alt")

        with ui.column(align_items="center").classes("w-[40vw] h-full"):
            files = (
                ui.table(
                    columns=[{"name": "path", "label": "File", "field": "path"}],
                    rows=[],
                )
                .classes("w-full h-[60vh]")
                .props("virtual-scroll")
            )

            with ui.row().classes("justify-between w-full"):
                ui.button(icon="delete_forever", on_click=reset_files).classes(
                    "w-max mt-4 text-bold bg-red"
                ).props("fab").tooltip("clear image buffer")
                ui.button(
                    icon="model_training",
                    on_click=train_model,
                ).classes("w-max mt-4 text-bold bg-green").props("fab").tooltip(
                    "start training"
                )

    with ui.page_sticky(position="bottom-left", x_offset=18, y_offset=18):
        ui.button(icon="bolt", on_click=hard_reset).classes("bg-red").props(
            "fab"
        ).tooltip("reset model")

    def update_page(result: str) -> None:
        image_src["source"] = result

    tmp_data.mkdir(parents=True, exist_ok=True)
    for file in tmp_data.iterdir():
        files.add_row({"path": file.name})

    await app_controller.set_handler(image_processor, update_page)
