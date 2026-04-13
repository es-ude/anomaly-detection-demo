import os

from nicegui import ui

from demo.web_app.controller.image_processing import BasicProcessor

from .layout import page_layout

FLIP_HORIZONTAL = os.getenv("FLIP_HORIZONTAL", "").lower() in ["true", "1"]
FLIP_VERTICAL = os.getenv("FLIP_VERTICAL", "").lower() in ["true", "1"]

basic_processor = BasicProcessor(
    target_image_size=(800, 800),
    flip_horizontal=FLIP_HORIZONTAL,
    flip_vertical=FLIP_VERTICAL,
)


async def basic(app_controller):
    page_layout()

    with ui.column().classes("w-full items-center"):
        result_image = ui.interactive_image().style(
            "object-fit: contain; height: 75vh; width: 75vh;"
        )

    def update_images(result: str) -> None:
        result_image.set_source(result)

    await app_controller.set_handler(basic_processor, update_images)
