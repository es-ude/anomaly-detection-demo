import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import os
    from functools import partial
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from demo.anomaly_detection.datasets.cookie_clf import CookieClfDataset
    from demo.anomaly_detection.model import Autoencoder
    from demo.anomaly_detection.persistence import load_model
    from demo.anomaly_detection.preprocessing import InferencePreprocessing

    DATASET_DIR = Path(os.environ["COOKIE_CLF_DATASET_DIR"])
    CKPT_DIR = Path(os.environ["COOKIE_CKPT_DIR"])
    AE_MODEL_FILE = CKPT_DIR / "ae_model.pt"


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
    return (ds_test,)


@app.cell
def _():
    model = Autoencoder()
    load_model(model, AE_MODEL_FILE)
    _ = model.eval()
    return (model,)


@app.cell
def _(ds_test, model):
    images, labels = ds_test[:]

    with torch.no_grad():
        encoded_images, reconstructed_images = model(images)
        predicted_labels = model.classify(images, reconstructed_images)

    images = images.squeeze(dim=-3)
    reconstructed_images = reconstructed_images.squeeze(dim=-3)

    accuracy = (predicted_labels == labels).sum() / len(labels)
    accuracy
    return images, predicted_labels, reconstructed_images


@app.cell
def _(ds_test):
    img_idx = mo.ui.number(start=0, stop=len(ds_test) - 1, label="Image Index")
    img_idx
    return (img_idx,)


@app.cell
def _(images, img_idx, predicted_labels, reconstructed_images):
    def plot_reconstructed(img_idx: int) -> plt.Figure:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), tight_layout=True)
        axs[0].imshow(images[img_idx], vmin=0, vmax=1, cmap="gray")
        axs[1].imshow(reconstructed_images[img_idx], vmin=0, vmax=1, cmap="gray")
        fig.suptitle("Normal" if predicted_labels[img_idx] == 0 else "Defect")
        return fig

    plot_reconstructed(img_idx.value)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
