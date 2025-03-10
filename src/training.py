from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.in_memory_dataset import load_in_memory_dataset


def train_autoencoder(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_test: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int = 0,
    in_memory_dataset: bool = True,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    history = dict(epoch=[], train_reconst_mse=[], test_reconst_mse=[])

    if in_memory_dataset:
        ds_train = load_in_memory_dataset(ds_train)
        ds_test = load_in_memory_dataset(ds_test)

    data_loader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)
    dl_train = data_loader(ds_train, shuffle=True)
    dl_test = data_loader(ds_test, shuffle=False)

    num_samples_train = len(dl_train)
    num_samples_test = len(dl_test)

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(1, epochs + 1):
        history["epoch"].append(epoch)

        model.train()
        running_loss = 0.0

        for input, _ in dl_train:
            input = input.to(device)

            model.zero_grad()
            reconstructed = model(input)
            loss = loss_fn(reconstructed, input)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * (len(input) / num_samples_train)

        history["train_reconst_mse"].append(running_loss)

        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for input, _ in dl_test:
                input = input.to(device)

                reconstructed = model(input)
                loss = loss_fn(reconstructed, input)
                running_loss += loss.item() * (len(input) / num_samples_test)

        history["test_reconst_mse"].append(running_loss)

        print(
            f"[{history['epoch'][-1]}/{epochs}] "
            f"train_reconst_mse: {history['train_reconst_mse'][-1]:.04f} ; "
            f"test_reconst_mse: {history['test_reconst_mse'][-1]:.04f}"
        )

    return history
