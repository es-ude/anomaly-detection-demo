import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import os
    from pathlib import Path
    from functools import partial
    from typing import Any, cast

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    from src.anomaly_detection.datasets.cookie_ad import CookieAdDataset
    from src.anomaly_detection.model import Autoencoder
    from src.anomaly_detection.persistence import load_model
    from src.anomaly_detection.preprocessing import InferencePreprocessing

    DATASET_DIR = Path(os.environ["COOKIE_AE_DATASET_DIR"])
    CKPT_DIR = Path(os.environ["COOKIE_CKPT_DIR"])
    MODEL_FILE = CKPT_DIR / "ae_model.pt"
    HISTORY_FILE = CKPT_DIR / "ae_history.csv"


@app.cell
def _():
    df_history = pd.read_csv(HISTORY_FILE)
    df_history[df_history["epoch"] > 5].plot(x="epoch", y=["train_reconst_mse", "test_reconst_mse"])
    return


@app.cell
def _():
    def binary_labels(label: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0) if label == 0 else torch.tensor(1)

    create_ds = partial(
            CookieAdDataset,
            dataset_dir=DATASET_DIR,
            sample_transform=InferencePreprocessing(
                target_img_width=int(os.environ["IMAGE_WIDTH"]),
                target_img_height=int(os.environ["IMAGE_HEIGHT"]),
            ),
            target_transform=binary_labels,
        )
    ds_train, ds_test = create_ds(training_set=True), create_ds(training_set=False)

    model = Autoencoder()
    load_model(model, MODEL_FILE)
    _ = model.eval()
    return ds_test, ds_train, model


@app.cell
def _(ds_test, ds_train, model):
    data: dict[str, Any] = {"set": [], "idx": [], "loss": [], "label": []}

    def perform_inference(ds: CookieAdDataset, set_name: str) -> None:
        loss_fn = torch.nn.L1Loss()

        for idx, (original, label) in enumerate(ds):
            with torch.no_grad():
                encoded, reconstructed = model(original)
                loss = loss_fn(reconstructed, original)

                data["set"].append(set_name)
                data["idx"].append(idx)
                data["loss"].append(loss.item())
                data["label"].append(int(label))

    perform_inference(ds_train, "train")
    perform_inference(ds_test, "test")

    df = pd.DataFrame.from_dict(data)
    df[["set", "loss"]].groupby("set").describe().round(decimals=4)
    return (df,)


@app.cell
def _(ds_test):
    img_idx = mo.ui.number(start=0, stop=len(ds_test)-1, label="Image Index")
    img_idx
    return (img_idx,)


@app.cell
def _(ds_test, img_idx, model):
    img_original, _ = cast(torch.Tensor, ds_test[img_idx.value])
    with torch.no_grad():
        img_reconstr = cast(torch.Tensor, model(img_original)[1])

    img_diff = (img_original - img_reconstr).abs()

    def plot_image(ax, img: torch.Tensor, title: str, cmap: Any, alpha=1) -> None:
        ax.imshow(img.movedim(0, -1), cmap=cmap, vmin=0, vmax=1, alpha=alpha)
        ax.set_title(title)

    def plot_combined(ax) -> None:
        original = (img_original.movedim(0, -1).numpy() * 255).astype(np.uint8)
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        diff = (img_diff.movedim(0, -1).numpy() * 255).astype(np.uint8)
        diff = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        combined = cv2.addWeighted(original, 0.4, diff, 0.6, 0)
        ax.imshow(combined)
        ax.set_title("Superimposed")

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), tight_layout=True)
    plot_image(axs[0, 0], img_original, title="Original", cmap="gray")
    plot_image(axs[0, 1], img_reconstr, title="Reconstructed", cmap="gray")
    plot_image(axs[1, 0], img_diff, title="Difference", cmap="hot")
    plot_combined(axs[1, 1])
    fig
    return


@app.cell
def _(df):
    decision_boundary = df.loc[df["set"] == "train", "loss"].quantile(0.95)

    df["prediction"] = 0
    df.loc[df["loss"] > decision_boundary, "prediction"] = 1

    df_test = df[df["set"] == "test"]

    accuracy = sum(df_test["label"] == df_test["prediction"]) / len(df_test)
    accuracy
    return decision_boundary, df_test


@app.cell
def _(decision_boundary, df_test):
    plt.hist(df_test[df_test["label"] == 0]["loss"], color="tab:green", alpha=0.8)
    plt.hist(df_test[df_test["label"] == 1]["loss"], color="tab:blue", alpha=0.8)
    plt.axvline(x=decision_boundary, color="r", linestyle="--")
    return


if __name__ == "__main__":
    app.run()
