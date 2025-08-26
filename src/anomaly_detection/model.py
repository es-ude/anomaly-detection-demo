import torch


class Encoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_conv_block(in_channels=1, out_channels=32),  # 64
            *_conv_block(in_channels=32, out_channels=64),  # 32
            *_conv_block(in_channels=64, out_channels=128),  # 16
            *_conv_block(in_channels=128, out_channels=128),  # 8
        )


class Decoder(torch.nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            *_deconv_block(in_channels=128, out_channels=128),  # 16
            *_deconv_block(in_channels=128, out_channels=64),  # 32
            *_deconv_block(in_channels=64, out_channels=32),  # 64
            *_deconv_block(in_channels=32, out_channels=32),  # 128
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),  # 128
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
            kernel_size=5,
            stride=2,
            padding=2,
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


class Classifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128, out_channels=2, kernel_size=3, padding="same"
            ),
            torch.nn.MaxPool2d(kernel_size=8),
            torch.nn.Flatten(start_dim=-3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent_space = self.encoder(inputs)
        return self.classifier(latent_space)


class CookieAdModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.autoencoder = Autoencoder()
        self.classifier = Classifier(self.autoencoder.encoder)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_space = self.autoencoder.encoder(inputs)
        reconstructed = self.autoencoder.decoder(latent_space)
        prediction = self.classifier(latent_space)
        return reconstructed, prediction

    def from_models(self, autoencoder: Autoencoder, classifier: Classifier) -> None:
        self.autoencoder = autoencoder
        self.classifier = classifier.classifier
