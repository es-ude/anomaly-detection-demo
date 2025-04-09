from collections.abc import Callable
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset


def train_autoencoder(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_test: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0,
    augment_input_image: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    history = dict(epoch=[], train_reconst_mse=[], test_reconst_mse=[])

    data_loader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)
    dl_train = data_loader(ds_train, shuffle=True)
    dl_test = data_loader(ds_test, shuffle=False)

    num_samples_train = len(dl_train)
    num_samples_test = len(dl_test)

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.to(device)

    for epoch in range(1, epochs + 1):
        history["epoch"].append(epoch)

        model.train()
        running_loss = 0.0

        for original_input, _ in dl_train:
            augmented_input = augment_input_image(original_input).to(device)
            original_input = original_input.to(device)

            model.zero_grad()
            reconstructed = model(augmented_input)
            loss = loss_fn(reconstructed, original_input)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * (len(original_input) / num_samples_train)

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
