import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
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
    from sklearn.decomposition import PCA

    from src.anomaly_detection.datasets.cookie_ad import CookieAD
    from src.anomaly_detection.model import Autoencoder
    from src.anomaly_detection.persistence import load_model
    from src.anomaly_detection.preprocessing import InferencePreprocessing

    DATASET_DIR = Path(os.environ["COOKIE_DATASET_DIR"])
    OUTPUT_DIR = Path(os.environ["COOKIE_OUTPUT_DIR"])

    MODEL_FILE = OUTPUT_DIR / os.environ["MODEL_FILE_NAME"]
    HISTORY_FILE = OUTPUT_DIR / os.environ["HISTORY_FILE_NAME"]
    return (
        Any,
        Autoencoder,
        CookieAD,
        DATASET_DIR,
        InferencePreprocessing,
        MODEL_FILE,
        PCA,
        load_model,
        mo,
        np,
        os,
        partial,
        pd,
        plt,
        torch,
    )


@app.cell
def _(CookieAD, DATASET_DIR, InferencePreprocessing, os, partial, torch):
    def binary_labels(label: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0) if label == 0 else torch.tensor(1)

    create_ds = partial(
        CookieAD,
        dataset_dir=DATASET_DIR,
        sample_transform=InferencePreprocessing(
            target_img_width=int(os.environ["IMAGE_WIDTH"]),
            target_img_height=int(os.environ["IMAGE_HEIGHT"]),
        ),
        target_transform=binary_labels,
    )
    ds_train, ds_test = create_ds(training_set=True), create_ds(training_set=False)
    return ds_test, ds_train


@app.cell
def _(Autoencoder, MODEL_FILE, load_model):
    model = Autoencoder()
    load_model(model, MODEL_FILE)
    _ = model.eval()
    return (model,)


@app.cell
def _(Any, CookieAD, ds_test, ds_train, model, pd, torch):
    data: dict[str, Any] = {"set": [], "idx": [], "loss": [], "label": [], "encoded": []}

    def perform_inference(ds: CookieAD, set_name: str) -> None:
        loss_fn = torch.nn.L1Loss()

        for idx, (original, label) in enumerate(ds):
            with torch.no_grad():
                encoded = model.encoder(original)
                reconstructed = model.decoder(encoded)
                loss = loss_fn(reconstructed, original)

                data["set"].append(set_name)
                data["idx"].append(idx)
                data["loss"].append(loss.item())
                data["label"].append(int(label))
                data["encoded"].append(encoded.numpy())

    perform_inference(ds_train, "train")
    perform_inference(ds_test, "test")

    df = pd.DataFrame.from_dict(data)

    print(df[["set", "loss"]].groupby("set").describe())
    return (df,)


@app.cell
def _():
    128*128*1
    return


@app.cell
def _(PCA, df, mo, np, plt):
    latent_vectors = np.stack(df["encoded"]).reshape(len(df), -1)

    pca = PCA(n_components=3)
    cluster = pca.fit_transform(latent_vectors)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(cluster[:,0], cluster[:,1], cluster[:,2], c=df["label"])

    mo.mpl.interactive(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
