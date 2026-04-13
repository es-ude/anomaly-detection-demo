import asyncio
import os
from pathlib import Path

from nicegui import app, ui

from demo.web_app.controller.demo_application_controller import (
    DemoApplicationController,
)
from demo.web_app.pages import register_pages

USE_PICAM_MODULE = os.getenv("ENABLE_PICAM", "").lower() in ["true", "1"]
CAM_PORT = int(os.getenv("CAM_PORT", 0))

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
