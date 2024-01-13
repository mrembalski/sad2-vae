# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, line-too-long, too-many-arguments
from torch import nn
from typeguard import typechecked

class ResNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, output_padding=0):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.LeakyReLU(inplace=True),
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
        x = nn.LeakyReLU(inplace=True)(x)
        return x

class Decoder(nn.Module):
    @typechecked
    def __init__(self, stride: int, output_channels: int, encoder_hidden_dims: list[int], latent_space_dim: int, flattened_size: int, reduced_size: int):
        super().__init__()
        print("Decoder:")
        hidden_dims = encoder_hidden_dims[::-1]
        self.hidden_dims = hidden_dims
        self.reduced_size = reduced_size
        self.flattened_size = flattened_size

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_space_dim, self.flattened_size),
            nn.LeakyReLU(),
        )
        print(f"\t{latent_space_dim} -> {self.flattened_size}")

        output_padding = 1 if stride > 1 else 0
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(ResNetDecoderBlock(hidden_dims[i], hidden_dims[i + 1], stride=stride, output_padding=output_padding))
            print(f"\t{hidden_dims[i]} -> {hidden_dims[i + 1]}")

        self.decoder = nn.Sequential(*modules)

        # Final layer adjusted to output the correct number of channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], output_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.Sigmoid(),
        )
        print(f"\t{hidden_dims[-1]} -> {output_channels}")

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], self.reduced_size, self.reduced_size)
        x = self.decoder(x)
        result = self.final_layer(x)
        return result
