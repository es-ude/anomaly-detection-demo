import torch
from torch.nn import Conv2d, ConvTranspose2d, Flatten, Linear, ReLU, Unflatten


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(in_channels=64, out_channels=128, kernel_size=8, stride=4),
            ReLU(),
            Flatten(),
            Linear(in_features=512, out_features=128),
            ReLU(),
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            Linear(in_features=128, out_features=512),
            ReLU(),
            Unflatten(dim=1, unflattened_size=(128, 2, 2)),
            ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=8,
                stride=4,
                output_padding=2,
            ),
            ReLU(),
            ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=8,
                stride=4,
                output_padding=3,
            ),
            ReLU(),
            ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=8, stride=4),
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
