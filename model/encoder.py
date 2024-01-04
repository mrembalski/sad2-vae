# pylint: disable=missing-function-docstring, missing-module-docstring, line-too-long, missing-class-docstring
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_channels, encoder_hidden_dims):
        super().__init__()
        modules = []

        # use enumerate
        for h_dim in encoder_hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                ),
            )
            input_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        print(f"Encoder number of layers: {len(encoder_hidden_dims)}")

    def forward(self, x):
        return self.encoder(x)
