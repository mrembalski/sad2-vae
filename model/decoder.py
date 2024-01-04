# pylint: disable=missing-function-docstring, missing-module-docstring, line-too-long, missing-class-docstring
from torch import nn
from typeguard import typechecked


class Decoder(nn.Module):
    @typechecked
    def __init__(self, 
            initial_image_size: int,
            output_channels: int,
            encoder_hidden_dims: list[int],
            latent_space_dim: int,
        ):
        super().__init__()
        # Reversing the hidden dimensions for the decoder
        encoder_hidden_dims.reverse()
        hidden_dims = encoder_hidden_dims
        self.hidden_dims = hidden_dims

        # Calculate the reduced size based on the number of conv layers
        self.reduced_size = initial_image_size // (2 ** len(hidden_dims))

        # Calculate the flattened size based on the last layer of conv dimensions
        self.flattened_size = self.hidden_dims[0] * (self.reduced_size ** 2)

        # Linear layer to expand from the latent space with proper activation
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_space_dim, self.flattened_size),
            nn.LeakyReLU(),
        )

        # Transposed convolutional layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                ),
            )

        self.decoder = nn.Sequential(*modules)

        # Final layer to get back to original image size and channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

        print(f"Decoder number of layers {len(self.decoder) + 1}")

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], self.reduced_size, self.reduced_size)
        x = self.decoder(x)
        result = self.final_layer(x)
        return result
