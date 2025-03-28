# Anomaly Detection Demo

## Datasets

The MVTec AD dataset can be downloaded [[here]](https://www.mvtec.com/company/research/datasets/mvtec-ad). The cookie dataset is only available internally. Please contact @julianhoever or @florianhettstedt

## Environment

Example environment file (`.env`) to set all required environment variables:

```bash
DEVICE=<e.g. cuda, mps or cpu>
NUM_WORKERS=0

IMAGE_WIDTH=128
IMAGE_HEIGHT=128

MVTEC_DATASET_DIR=<path to mvtec dataset>
MVTEC_OBJECT=hazelnut

MVTEC_SAVED_MODEL=<path to mvtec outputs>/model.pt
MVTEC_HISTORY_FILE=<path to mvtec outputs>/history.csv
MVTEC_VERSION_FILE=outputs/mvtec<path to mvtec outputs>/commit_hash.txt

COOKIE_DATASET_DIR=<path to cookie dataset>

COOKIE_SAVED_MODEL=<path to cookie outputs>/model.pt
COOKIE_HISTORY_FILE=<path to cookie outputs>/history.csv
COOKIE_VERSION_FILE=<path to cookie outputs>/commit_hash.txt
```

## Run Experiments

### MVTec (Hazelnut) Training
```bash
uv run --env-file=.env experiments/train_mvtec_autoencoder.py
```

### Cookie Training
```bash
uv run --env-file=.env experiments/train_cookie_autoencoder.py
```
