import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import os
    from functools import partial
    from pathlib import Path

    import torch
    import matplotlib.pyplot as plt

    from src.anomaly_detection.datasets.cookie_clf import CookieClfDataset
    from src.anomaly_detection.model import Autoencoder, Classifier
    from src.anomaly_detection.persistence import load_model
    from src.anomaly_detection.preprocessing import InferencePreprocessing

    DATASET_DIR = Path(os.environ["COOKIE_CLF_DATASET_DIR"])
    CKPT_DIR = Path(os.environ["COOKIE_CKPT_DIR"])
    AE_MODEL_FILE = CKPT_DIR / "ae_model.pt"
    CLF_MODEL_FILE = CKPT_DIR / "clf_model.pt"


@app.cell
def _():
    create_ds = partial(
        CookieClfDataset,
        dataset_dir=DATASET_DIR,
        sample_transform=InferencePreprocessing(
            target_img_width=int(os.environ["IMAGE_WIDTH"]),
            target_img_height=int(os.environ["IMAGE_HEIGHT"]),
        ),
    )
    ds_train, ds_test = create_ds(training_set=True), create_ds(training_set=False)
    return ds_test, ds_train


@app.cell
def _():
    ae_model = Autoencoder()
    load_model(ae_model, AE_MODEL_FILE)
    _ = ae_model.eval()

    clf_model = Classifier()
    load_model(clf_model, CLF_MODEL_FILE)
    _ = clf_model.eval()
    return ae_model, clf_model


@app.cell
def _(ae_model, clf_model, ds_train):
    images, labels = ds_train[:]

    with torch.no_grad():
        encoded_images, reconstructed_images = ae_model(images)
        predicted_labels = clf_model(encoded_images)

    images = images.squeeze(dim=-3)
    reconstructed_images = reconstructed_images.squeeze(dim=-3)
    predicted_labels = predicted_labels.argmax(dim=-1)

    accuracy = (predicted_labels == labels).sum() / len(labels)
    accuracy
    return images, reconstructed_images


@app.cell
def _(ds_test):
    img_idx = mo.ui.number(start=0, stop=len(ds_test), label="Image Index")
    img_idx
    return (img_idx,)


@app.cell
def _(images, img_idx, reconstructed_images):
    def plot_reconstructed(img_idx: int) -> plt.Figure:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(images[img_idx], vmin=0, vmax=1, cmap="gray")
        axs[1].imshow(reconstructed_images[img_idx], vmin=0, vmax=1, cmap="gray")
        return fig

    plot_reconstructed(img_idx.value)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
