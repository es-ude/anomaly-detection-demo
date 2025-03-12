# Anomaly Detection Demo

Example environment file (`.env`) to set all required environment variables:

```bash
SAVED_MODEL=<path-to-output-directory>/model.pt
HISTORY_FILE=<path-to-output-directory>/history.csv
VERSION_FILE=<path-to-output-directory>/commit_hash.txt

DEVICE=<e.g. cuda or mps>
NUM_WORKERS=0

IMAGE_WIDTH=128
IMAGE_HEIGHT=128

MVTEC_DATASET_DIR=<path-to-mvtec-ad-dataset>
MVTEC_OBJECT=hazelnut

COOKIE_DATASET_DIR=<path-to-cookie-dataset>
```

To start the training:

```bash
uv run --env-file=.env experiments/train_mvtec_autoencoder.py
```