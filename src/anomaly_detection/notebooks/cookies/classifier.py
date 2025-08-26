import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import os
    from functools import partial
    from pathlib import Path
    from typing import Any, cast

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.decomposition import PCA

    from src.anomaly_detection.datasets.cookie_clf import CookieClfDataset
    from src.anomaly_detection.model import Autoencoder
    from src.anomaly_detection.persistence import load_model
    from src.anomaly_detection.preprocessing import InferencePreprocessing
    from src.anomaly_detection.training import train_classifier

    DATASET_DIR = Path(os.environ["COOKIE_CLF_DATASET_DIR"])
    OUTPUT_DIR = Path(os.environ["COOKIE_OUTPUT_DIR"])

    AE_MODEL_FILE = OUTPUT_DIR / os.environ["MODEL_FILE_NAME"]
    HISTORY_FILE = OUTPUT_DIR / os.environ["HISTORY_FILE_NAME"]
    return (
        AE_MODEL_FILE,
        Any,
        Autoencoder,
        CookieClfDataset,
        DATASET_DIR,
        InferencePreprocessing,
        PCA,
        load_model,
        mo,
        np,
        os,
        partial,
        pd,
        plt,
        torch,
        train_classifier,
    )


@app.cell
def dataset_loading(
    CookieClfDataset,
    DATASET_DIR,
    InferencePreprocessing,
    os,
    partial,
):
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
def ae_loading(AE_MODEL_FILE, Autoencoder, load_model):
    autoencoder = Autoencoder()
    load_model(autoencoder, AE_MODEL_FILE)
    _ = autoencoder.eval()
    return (autoencoder,)


@app.cell
def ae_inference(
    Any,
    CookieClfDataset,
    autoencoder,
    ds_test,
    ds_train,
    pd,
    torch,
):
    data: dict[str, Any] = {
        "set": [],
        "idx": [],
        "loss": [],
        "label": [],
        "encoded": [],
    }

    def perform_inference(ds: CookieClfDataset, set_name: str) -> None:
        loss_fn = torch.nn.L1Loss()

        for idx, (original, label) in enumerate(ds):
            with torch.no_grad():
                encoded = autoencoder.encoder(original)
                reconstructed = autoencoder.decoder(encoded)
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
def latent_space(PCA, df, mo, np, plt):
    df_selected = df[df["set"] == "train"]
    latent_vectors = np.stack(df_selected["encoded"]).reshape(len(df_selected), -1)

    pca = PCA(n_components=3)
    cluster = pca.fit_transform(latent_vectors)

    def plot_latent_space() -> plt.Figure:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=df_selected["label"])
        return fig

    mo.mpl.interactive(plot_latent_space())
    return


@app.cell
def clf_definition(Autoencoder, torch):
    class Classifier(torch.nn.Module):
        def __init__(self, autoencoder: Autoencoder) -> None:
            super().__init__()
            self.autoencoder = autoencoder.eval()

            self.classifier = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding="same"),
                torch.nn.MaxPool2d(kernel_size=8),
                torch.nn.Flatten(start_dim=-3),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                latent_space = self.autoencoder.encoder(inputs)        
            return self.classifier(latent_space)
    return (Classifier,)


@app.cell
def train_clf(
    Classifier,
    autoencoder,
    ds_test,
    ds_train,
    mo,
    plt,
    train_classifier,
):
    classifier = Classifier(autoencoder)
    print("Params:", sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    history = train_classifier(
        model=classifier,
        ds_train=ds_train,
        ds_test=ds_test,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
    )

    def plot_history() -> plt.Figure:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(history["epoch"], history["train_loss"], label="train loss")
        ax.plot(history["epoch"], history["test_loss"], label="test loss")
        ax.legend()
        return fig

    mo.mpl.interactive(plot_history())
    return (classifier,)


@app.cell
def compute_accuracy(classifier, ds_test):
    samples, labels = ds_test[:]
    predictions = classifier(samples).argmax(dim=-1)

    (predictions == labels).sum() / len(ds_test)
    return


if __name__ == "__main__":
    app.run()
