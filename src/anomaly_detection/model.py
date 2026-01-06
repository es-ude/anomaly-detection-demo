from typing import Optional

import torch
from torch.nn.functional import mse_loss


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_conv_block(in_channels=1, out_channels=32),  # 256
            *_conv_block(in_channels=32, out_channels=64),  # 128
            *_conv_block(in_channels=64, out_channels=128),  # 64
            *_conv_block(in_channels=128, out_channels=128),  # 32
            torch.nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1),  # 32
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            torch.nn.Conv2d(in_channels=8, out_channels=128, kernel_size=1),  # 32
            *_deconv_block(in_channels=128, out_channels=128),  # 64
            *_deconv_block(in_channels=128, out_channels=64),  # 128
            *_deconv_block(in_channels=64, out_channels=32),  # 256
            *_deconv_block(in_channels=32, out_channels=32),  # 512
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),  # 512
        )


class Autoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.register_buffer(
            "decision_boundary", torch.tensor(-1, dtype=torch.get_default_dtype())
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded

    @torch.no_grad
    def determine_decision_boundary(
        self, calibration_data: torch.Tensor, quantile: float
    ) -> None:
        _, reconstructed_data = self(calibration_data)
        deviations = _compute_deviation(reconstructed_data, calibration_data)
        self.decision_boundary = deviations.quantile(quantile)

    @torch.no_grad
    def classify(
        self, inputs: torch.Tensor, reconstructed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.decision_boundary == -1:
            raise ValueError(
                "Invalid `decision_boundary`. Call `determine_decision_boundary(...)` before `classify(...)`."
            )

        if reconstructed is None:
            _, reconst = self(inputs)
        else:
            reconst = reconstructed

        deviation = _compute_deviation(inputs, reconst)

        return (deviation > self.decision_boundary).to(torch.long)


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


def _compute_deviation(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> torch.Tensor:
    return mse_loss(reconstructed, original, reduction="none").mean(dim=[-1, -2, -3])
