#!/usr/bin/env bash

## Autostart script
# This script is used to automatically start the demo on boot.
# It relies on the already set-up repositury and virutual environment, as well as additonal programs.
# - astrl-uv
# - chromium browser

## How to?
# 2. Add this script to '.config/labwc/autostart'
# 3. Reboot machine

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}" || exit 1

export ENABLE_PI_CAM=1
uv run src/demo_interface/demo.py &
sleep 30
chromium http://127.0.0.1:8080/anomaly-detection --kiosk --noerrdialogs --disable-infobars --no-first-run --enable-features=OverlayScrollbar --start-maximized --force-device-scale-factor=1.2
