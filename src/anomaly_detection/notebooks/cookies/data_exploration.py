import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import os
    from pathlib import Path
    from functools import partial

    import matplotlib.pyplot as plt

    from src.anomaly_detection.datasets.cookie_ad import CookieAdDataset
    from src.anomaly_detection.preprocessing import InferencePreprocessing


@app.cell
def _():
    cookie_dataset = partial(
        CookieAdDataset,
        dataset_dir=Path(os.environ["COOKIE_AE_DATASET_DIR"]),
        in_memory=False,
        sample_transform=InferencePreprocessing(
            target_img_width=int(os.environ["IMAGE_WIDTH"]),
            target_img_height=int(os.environ["IMAGE_HEIGHT"]),
        ),
    )

    ds_train = cookie_dataset(training_set=True)
    ds_test = cookie_dataset(training_set=False)

    print(f"Train Length: {len(ds_train)}")
    print(f"Test Length: {len(ds_test)}")
    print(f"Image Shape: {ds_train[0][0].shape}")
    return (ds_test,)


@app.cell
def _(ds_test):
    img_idx = mo.ui.number(start=0, stop=len(ds_test)-1, label="Image Index")
    img_idx
    return (img_idx,)


@app.cell
def _(ds_test, img_idx):
    image, label = ds_test[img_idx.value]
    print(image.shape)
    print(label)
    plt.imshow(image.movedim(0, -1), vmin=0, vmax=1, cmap="gray")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
