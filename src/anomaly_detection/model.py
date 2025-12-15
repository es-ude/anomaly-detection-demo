import torch


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_conv_block(in_channels=1, out_channels=16),  # 256
            *_conv_block(in_channels=16, out_channels=32),  # 128
            *_conv_block(in_channels=32, out_channels=64),  # 64
            *_conv_block(in_channels=64, out_channels=128),  # 32
            *_conv_block(in_channels=128, out_channels=128),  # 16
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_deconv_block(in_channels=128, out_channels=128),  # 32
            *_deconv_block(in_channels=128, out_channels=64),  # 64
            *_deconv_block(in_channels=64, out_channels=32),  # 128
            *_deconv_block(in_channels=32, out_channels=16),  # 256
            *_deconv_block(in_channels=16, out_channels=16),  # 512
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),  # 512
        )


class Autoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Classifier(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_dwsep_conv(
                in_channels=128, out_channels=16, kernel_size=3, padding="same"
            ),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),
            *_dwsep_conv(in_channels=16, out_channels=2, kernel_size=3, padding="same"),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(start_dim=-3),
        )


def _dwsep_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int | str = 0,
) -> list[torch.nn.Module]:
    return [
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        ),
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        ),
    ]


def _conv_block(in_channels: int, out_channels: int) -> list[torch.nn.Module]:
    return [
        *_dwsep_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=2,
            padding=2,
        ),
        torch.nn.ReLU(),
    ]


def _deconv_block(in_channels: int, out_channels: int) -> list[torch.nn.Module]:
    return [
        torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            groups=in_channels,
        ),
        torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
        ),
        torch.nn.ReLU(),
    ]
