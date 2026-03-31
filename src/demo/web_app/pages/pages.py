from nicegui import ui

from .anomaly_detection import anomaly_detection
from .basic import basic
from .calibration import calibrate

__app_controller = None


async def basic_page():
    global __app_controller
    await basic(__app_controller)


async def calibrate_page():
    global __app_controller
    await calibrate(__app_controller)


async def anomaly_detection_page():
    global __app_controller
    await anomaly_detection(__app_controller)


async def register_pages(app_controller):
    global __app_controller
    __app_controller = app_controller

    ui.page("/", title="Home")(lambda: ui.navigate.to("/anomaly-detection"))
    ui.page("/basic", title="Basic Image Processing")(basic_page)
    ui.page("/calibration", title="Calibration")(calibrate_page)
    ui.page("/anomaly-detection", title="Anomaly Detection")(anomaly_detection_page)
