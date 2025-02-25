import torch


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_conv_block(in_channels=3, out_channels=64, pool_size=4),  # 64
            *_conv_block(in_channels=64, out_channels=32),  # 32
            *_conv_block(in_channels=32, out_channels=1),  # 16
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_deconv_block(in_channels=1, out_channels=32, up_size=32),
            *_deconv_block(in_channels=32, out_channels=64, up_size=64),
            *_deconv_block(
                in_channels=64, out_channels=3, up_size=256, final_layer=True
            ),
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


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    pool_size: int = 2,
) -> list[torch.nn.Module]:
    return [
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=in_channels,
        ),
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        ),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(num_features=out_channels),
        torch.nn.MaxPool2d(kernel_size=pool_size),
    ]


def _deconv_block(
    in_channels: int,
    out_channels: int,
    up_size: int,
    kernel_size: int = 3,
    stride: int = 1,
    final_layer: bool = False,
) -> list[torch.nn.Module]:
    if final_layer:
        block_tail = [torch.nn.Sigmoid()]
    else:
        block_tail = [
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(),
        ]

    return [
        torch.nn.UpsamplingBilinear2d((up_size, up_size)),
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=in_channels,
        ),
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        ),
        *block_tail,
    ]
