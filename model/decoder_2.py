# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring, line-too-long, too-many-arguments
from torch import nn
from typeguard import typechecked

class ResNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, output_padding=0, expansion=4):
        super().__init__()
        self.expansion = expansion

        # Adjusted to start with expanded channel size and reduce it
        expanded_in_channels = in_channels
        expanded_out_channels = out_channels * self.expansion

        self.residual_function = nn.Sequential(
            nn.ConvTranspose2d(expanded_in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, expanded_out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or expanded_in_channels != expanded_out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(expanded_in_channels, expanded_out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding),
            )

        print(f"ResNetDecoderBlock: {in_channels} -> {out_channels} -> {expanded_out_channels}")

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual_function(x)

        x = residual + shortcut
        x = nn.ReLU(inplace=True)(x)
        return x

class Decoder(nn.Module):
    @typechecked
    def __init__(self, stride: int, output_channels: int, encoder_hidden_dims: list[int], latent_space_dim: int, expansion: int, flattened_size: int, reduced_size: int):
        super().__init__()

        hidden_dims = encoder_hidden_dims[::-1]
        self.hidden_dims = hidden_dims
        self.reduced_size = reduced_size
        self.flattened_size = flattened_size
        self.expansion = expansion

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_space_dim, self.flattened_size),
            nn.ReLU(),
        )

        output_padding = 1 if stride > 1 else 0
        modules = []
        for i in range(len(hidden_dims) - 1):
            # Adjusting in_channels to match the output of encoder's corresponding block
            modules.append(ResNetDecoderBlock(hidden_dims[i] * expansion, hidden_dims[i + 1], stride=stride, output_padding=output_padding, expansion=expansion))

        self.decoder = nn.Sequential(*modules)

        # Final layer adjusted to output the correct number of channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1] * expansion, output_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0] * self.expansion, self.reduced_size, self.reduced_size)
        x = self.decoder(x)
        result = self.final_layer(x)
        return result
