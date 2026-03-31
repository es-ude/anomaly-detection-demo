from nicegui import ui

from demo.web_app.controller.image_processing import CalibrationProcessor

from .layout import page_layout

calibration_processor = CalibrationProcessor(target_image_size=(800, 800))


async def calibrate(app_controller):
    page_layout()

    with ui.column().classes("w-full items-center"):
        result_image = ui.interactive_image().style(
            "object-fit: contain; height: 75vh; width: 75vh;"
        )

    def update_images(result: str) -> None:
        result_image.set_source(result)

    await app_controller.set_handler(calibration_processor, update_images)
