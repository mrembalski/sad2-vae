# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, line-too-long, too-many-arguments, too-few-public-methods
from torch import nn
from typeguard import typechecked

class ResNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, output_padding=0, add_relu=False):
        super().__init__()
        self.add_relu = add_relu

        self.residual_function = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding),
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual_function(x)
        x = residual + shortcut

        if self.add_relu:
            x = nn.ReLU()(x)

        return x

class Decoder(nn.Module):
    @typechecked
    def __init__(self, stride: int, output_channels: int, encoder_hidden_dims: list[int], latent_space_dim: int, flattened_size: int, reduced_size: int):
        super().__init__()
        hidden_dims = encoder_hidden_dims[::-1]
        self.hidden_dims = hidden_dims
        self.reduced_size = reduced_size
        self.flattened_size = flattened_size

        print("Decoder input:")
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_space_dim, self.flattened_size),
            nn.ReLU(),
        )
        print(f"\t{latent_space_dim} -> {self.flattened_size} (flattened)")

        print("Decoder:")
        output_padding = 1 if stride > 1 else 0
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            if i == len(hidden_dims) - 1:
                modules.append(ResNetDecoderBlock(h_dim, output_channels, stride=1, output_padding=0, add_relu=False))
                print(f"\t{h_dim} -> {output_channels}, stride=1")
            else:
                modules.append(ResNetDecoderBlock(h_dim, hidden_dims[i + 1], stride=stride, output_padding=output_padding, add_relu=True))
                print(f"\t{h_dim} -> {hidden_dims[i + 1]}, stride={stride}")

        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], self.reduced_size, self.reduced_size)
        result = self.decoder(x)
        return result
