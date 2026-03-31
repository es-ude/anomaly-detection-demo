import asyncio
import os
from pathlib import Path

from nicegui import app, ui

from demo.web_app.controller.demo_application_controller import (
    DemoApplicationController,
)
from demo.web_app.pages import register_pages

USE_PICAM_MODULE = "ENABLE_PI_CAM" in os.environ
CAM_PORT = int(os.environ.get("CAM_PORT", 0))
IMAGE_WIDTH = int(os.environ["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(os.environ["IMAGE_HEIGHT"])

PLACEHOLDER_IMAGE = Path(__file__).parent.joinpath("controller", "placeholder.png")


async def setup() -> None:
    app_controller = DemoApplicationController(
        cam_port=CAM_PORT,
        placeholder_image_file=PLACEHOLDER_IMAGE,
        use_picam=USE_PICAM_MODULE,
    )

    asyncio.create_task(app_controller.run())
    app.on_shutdown(app_controller.close)

    print("Visit your app on one of these URLs:", app.urls)
    await register_pages(app_controller)


if __name__ in {"__main__", "__mp_main__"}:
    app.on_startup(setup)
    ui.run(
        host="0.0.0.0",
        port=8080,
        title="CookieAdDemo",
        favicon="🍪",
        show=False,
        show_welcome_message=False,
    )
