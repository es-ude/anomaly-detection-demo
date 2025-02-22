from typing import Any

import torch
from torch.nn import Conv2d, ConvTranspose2d, Flatten, Linear, ReLU, Unflatten


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        conv_kwargs: dict[str, Any] = dict(kernel_size=5, stride=2)
        super().__init__(
            Conv2d(in_channels=3, out_channels=8, **conv_kwargs),
            ReLU(),
            Conv2d(in_channels=8, out_channels=16, **conv_kwargs),
            ReLU(),
            Conv2d(in_channels=16, out_channels=32, **conv_kwargs),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, **conv_kwargs),
            ReLU(),
            Flatten(),
            Linear(in_features=10816, out_features=256),
            ReLU(),
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        conv_kwargs: dict[str, Any] = dict(kernel_size=5, stride=2)
        super().__init__(
            Linear(in_features=256, out_features=10816),
            ReLU(),
            Unflatten(dim=1, unflattened_size=(64, 13, 13)),
            ConvTranspose2d(in_channels=64, out_channels=32, **conv_kwargs),
            ReLU(),
            ConvTranspose2d(in_channels=32, out_channels=16, **conv_kwargs),
            ReLU(),
            ConvTranspose2d(
                in_channels=16, out_channels=8, output_padding=1, **conv_kwargs
            ),
            ReLU(),
            ConvTranspose2d(
                in_channels=8, out_channels=3, output_padding=1, **conv_kwargs
            ),
            ReLU(),
        )


class Autoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded
