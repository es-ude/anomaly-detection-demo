import torch


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_conv_block(in_channels=3, out_channels=64),  # 128
            *_conv_block(in_channels=64, out_channels=128),  # 64
            *_conv_block(in_channels=128, out_channels=256),  # 32
            *_conv_block(in_channels=256, out_channels=512),  # 16
            *_conv_block(in_channels=512, out_channels=1024),  # 8
            *_conv_block(in_channels=1024, out_channels=2048),  # 4
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_deconv_block(in_channels=2048, out_channels=1024, up_size=8),
            *_deconv_block(in_channels=1024, out_channels=512, up_size=16),
            *_deconv_block(in_channels=512, out_channels=256, up_size=32),
            *_deconv_block(in_channels=256, out_channels=128, up_size=64),
            *_deconv_block(in_channels=128, out_channels=64, up_size=128),
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
        torch.nn.MaxPool2d(
            kernel_size=pool_size,
        ),
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
        torch.nn.Upsample(size=(up_size, up_size), mode="bilinear"),
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
