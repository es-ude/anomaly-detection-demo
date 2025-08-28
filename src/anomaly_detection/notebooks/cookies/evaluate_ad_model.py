import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from functools import partial
    from pathlib import Path

    import torch
    import matplotlib.pyplot as plt

    from src.anomaly_detection.datasets.cookie_clf import CookieClfDataset
    from src.anomaly_detection.model import CookieAdModel
    from src.anomaly_detection.persistence import load_model
    from src.anomaly_detection.preprocessing import InferencePreprocessing

    DATASET_DIR = Path(os.environ["COOKIE_CLF_DATASET_DIR"])
    CKPT_DIR = Path(os.environ["COOKIE_CKPT_DIR"])
    AD_MODEL_FILE = CKPT_DIR / "ad_model.pt"
    return (
        AD_MODEL_FILE,
        CookieAdModel,
        CookieClfDataset,
        DATASET_DIR,
        InferencePreprocessing,
        load_model,
        os,
        partial,
        plt,
        torch,
    )


@app.cell
def _(CookieClfDataset, DATASET_DIR, InferencePreprocessing, os, partial):
    create_ds = partial(
        CookieClfDataset,
        dataset_dir=DATASET_DIR,
        sample_transform=InferencePreprocessing(
            target_img_width=int(os.environ["IMAGE_WIDTH"]),
            target_img_height=int(os.environ["IMAGE_HEIGHT"]),
        ),
    )
    ds_train, ds_test = create_ds(training_set=True), create_ds(training_set=False)
    return (ds_train,)


@app.cell
def _(AD_MODEL_FILE, CookieAdModel, load_model):
    ad_model = CookieAdModel()
    load_model(ad_model, AD_MODEL_FILE)
    _ = ad_model.eval()
    return (ad_model,)


@app.cell
def _(ad_model, ds_train, torch):
    images, labels = ds_train[:]

    with torch.no_grad():
        reconstructed_images, predicted_labels = ad_model(images)

    images = images.squeeze(dim=-3)
    reconstructed_images = reconstructed_images.squeeze(dim=-3)
    predicted_labels = predicted_labels.argmax(dim=-1)
    return images, labels, predicted_labels, reconstructed_images


@app.cell
def _(images, plt, reconstructed_images):
    def plot_reconstructed(img_idx: int) -> plt.Figure:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(images[img_idx], cmap="gray")
        axs[1].imshow(reconstructed_images[img_idx], cmap="gray")
        return fig

    plot_reconstructed(123)
    return


@app.cell
def _(labels, predicted_labels):
    accuracy = (predicted_labels == labels).sum() / len(labels)
    accuracy
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
