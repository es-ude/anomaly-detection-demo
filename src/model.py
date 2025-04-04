import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            groups=in_channels,
        )
        self.pw_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.relu(x)
        return x


class DeConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw_conv = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            groups=in_channels,
        )
        self.pw_conv = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.relu(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ds1 = ConvBlock(in_channels=1, out_channels=32)  # 64
        self.ds2 = ConvBlock(in_channels=32, out_channels=64)  # 32
        self.ds3 = ConvBlock(in_channels=64, out_channels=128)  # 16
        self.ds4 = ConvBlock(in_channels=128, out_channels=128)  # 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.ds3(x)
        x = self.ds4(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.us1 = DeConvBlock(in_channels=128, out_channels=128)  # 16
        self.us2 = DeConvBlock(in_channels=128, out_channels=64)  # 32
        self.us3 = DeConvBlock(in_channels=64, out_channels=32)  # 64
        self.us4 = DeConvBlock(in_channels=32, out_channels=32)  # 128
        self.conv = torch.nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1
        )  # 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.us1(x)
        x = self.us2(x)
        x = self.us3(x)
        x = self.us4(x)
        x = self.conv(x)
        return x


class Autoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.dequant(x)
        return x
