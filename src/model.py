import torch


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_conv_block(in_channels=3, out_channels=64),  # 127
            *_conv_block(in_channels=64, out_channels=64),  # 63
            *_conv_block(in_channels=64, out_channels=128),  # 31
            *_conv_block(in_channels=128, out_channels=128),  # 15
            *_conv_block(in_channels=128, out_channels=256),  # 7
            *_conv_block(in_channels=256, out_channels=256),  # 3
            *_conv_block(in_channels=256, out_channels=32768),  # 1
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_deconv_block(in_channels=32768, out_channels=256, opad=0),
            *_deconv_block(in_channels=256, out_channels=256, opad=0),
            *_deconv_block(in_channels=256, out_channels=128, opad=0),
            *_deconv_block(in_channels=128, out_channels=128, opad=0),
            *_deconv_block(in_channels=128, out_channels=64, opad=0),
            *_deconv_block(in_channels=64, out_channels=64, opad=0),
            *_deconv_block(in_channels=64, out_channels=3, opad=1, final_layer=True),
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


def _conv_block(in_channels: int, out_channels: int) -> list[torch.nn.Module]:
    return [
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
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
    ]


def _deconv_block(
    in_channels: int, out_channels: int, opad: int, final_layer: bool = False
) -> list[torch.nn.Module]:
    return [
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        ),
        torch.nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            output_padding=opad,
            groups=out_channels,
        ),
        torch.nn.Sigmoid() if final_layer else torch.nn.ReLU(),
    ]
