#!/usr/bin/env bash

OUTPUT_BASE_DIR="$COOKIE_OUTPUT_DIR"
DATASET_BASE_DIR="data/CookieAD/v3"

echo "START"

echo
echo "=========="
echo

echo "run retrain cookie experiment:"
export COOKIE_OUTPUT_DIR="$OUTPUT_BASE_DIR/retrain/cookie"
export COOKIE_DATASET_DIR="$DATASET_BASE_DIR/cookie"
uv run src/demo/anomaly_detection/experiments/cookie/retrain_autoencoder.py

echo
echo "=========="
echo

echo "run retrain oreo experiment:"
export COOKIE_OUTPUT_DIR="$OUTPUT_BASE_DIR/retrain/oreo"
export COOKIE_DATASET_DIR="$DATASET_BASE_DIR/oreo"
uv run src/demo/anomaly_detection/experiments/cookie/retrain_autoencoder.py

echo
echo "=========="
echo

echo "run dataset size cookie experiment:"
export COOKIE_OUTPUT_DIR="$OUTPUT_BASE_DIR/ds_size/cookie"
export COOKIE_DATASET_DIR="$DATASET_BASE_DIR/cookie"
uv run src/demo/anomaly_detection/experiments/cookie/estimate_optimal_dataset.py

echo
echo "=========="
echo

echo "run dataset oreo experiment:"
export COOKIE_OUTPUT_DIR="$OUTPUT_BASE_DIR/ds_size/oreo"
export COOKIE_DATASET_DIR="$DATASET_BASE_DIR/oreo"
uv run src/demo/anomaly_detection/experiments/cookie/estimate_optimal_dataset.py

echo
echo "=========="
echo

echo "DONE"
