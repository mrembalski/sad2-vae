# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, line-too-long, too-few-public-methods
from torch import nn
from typeguard import typechecked

class ResNetEncoderBlock(nn.Module):
    @typechecked
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual_function(x)
        x = residual + shortcut
        x = nn.ReLU(inplace=True)(x)
        return x

class Encoder(nn.Module):
    @typechecked
    def __init__(self,
        input_channels: int,
        stride: int,
        encoder_hidden_dims: list[int],
    ):
        super().__init__()
        print("Encoder:")

        self.in_channels = input_channels
        layers = []

        for i, h_dim in enumerate(encoder_hidden_dims):
            if i == 0:
                layers.append(ResNetEncoderBlock(self.in_channels, h_dim, stride=1))
                print(f"\t{self.in_channels} -> {h_dim}, stride=1")
            else:
                layers.append(ResNetEncoderBlock(self.in_channels, h_dim, stride=stride))
                print(f"\t{self.in_channels} -> {h_dim}, stride={stride}")
            self.in_channels = h_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        result = self.encoder(x)
        return result
