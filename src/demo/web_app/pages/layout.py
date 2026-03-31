from pathlib import Path

from nicegui import ui

ASSETS_DIR = Path(__file__).parent.joinpath("assets")
ZAKID_LOGO = ASSETS_DIR / "zakid_logo_weiß.svg"
UDE_LOGO = ASSETS_DIR / "logo_ude_weiß_transparent.svg"


def page_layout():
    # set background of body to #020202
    ui.add_head_html("<style>body {background-color: #262626; }</style>")

    with ui.header(elevated=True).style("background-color: #262626"):
        with (
            ui.row()
            .classes("w-full items-center")
            .style("height: 5vh; min-height: 60px")
        ):
            ui.image(ZAKID_LOGO).props("width=200px height=50px fit=scale-down")
            ui.label("KI-basierte Defektkontrolle").classes(
                "flex-grow text-center text-4xl font-bold text-white"
            )
            ui.image(UDE_LOGO).props("width=200px height=50px fit=scale-down")
