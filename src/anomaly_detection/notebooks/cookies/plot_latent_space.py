import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import os
    from functools import partial
    from pathlib import Path

    import torch
    import plotly.express as px
    from sklearn.decomposition import PCA
    ''
    from src.anomaly_detection.datasets.cookie_clf import CookieClfDataset
    from src.anomaly_detection.model import Autoencoder
    from src.anomaly_detection.persistence import load_model
    from src.anomaly_detection.preprocessing import InferencePreprocessing

    DATASET_DIR = Path(os.environ["COOKIE_CLF_DATASET_DIR"])
    CKPT_DIR = Path(os.environ["COOKIE_CKPT_DIR"])
    AE_MODEL_FILE = CKPT_DIR / "ae_model.pt"


@app.cell
def dataset_loading():
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
def ae_loading():
    autoencoder = Autoencoder()
    load_model(autoencoder, AE_MODEL_FILE)
    _ = autoencoder.eval()
    return (autoencoder,)


@app.cell
def latent_space(autoencoder, ds_train):
    images, labels = ds_train[:]

    with torch.no_grad():
        latent_space = autoencoder.encoder(images)

    latent_vectors = latent_space.flatten(start_dim=-3)

    pca = PCA(n_components=3)
    vectors = pca.fit_transform(latent_vectors.numpy())

    mo.ui.plotly(
        px.scatter_3d(x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2], color=labels, opacity=0.7)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
