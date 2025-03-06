# Anomaly Detection Demo

Example environment file (`.env`) to set all required environment variables:

```bash
SAVED_MODEL=outputs/model.pt
HISTORY_FILE=outputs/history.csv
VERSION_FILE=outputs/commit_hash.txt

DEVICE=mps
NUM_WORKERS=0

AD_DATASET_DIR=<path_to_dataset>
AD_OBJECT=hazelnut

IMAGE_WIDTH=48
IMAGE_HEIGHT=48
```

To start the training:

```bash
uv run --env-file=.env src/train_autoencoder.py
```