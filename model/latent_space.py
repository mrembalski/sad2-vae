# pylint: disable=missing-function-docstring, missing-module-docstring, line-too-long, missing-class-docstring
from typeguard import typechecked
import torch
from torch import nn

class LatentSpace(nn.Module):
    """
    Simple LatentSpace class for the Hierarchical VAE.
    """

    @typechecked
    def __init__(self,
            initial_image_size: int,
            encoder_hidden_dims: list[int],
            dims: list[int],
        ):
        super().__init__()
        self.dims = dims

        # Calculate the reduced spatial dimensions
        self.reduced_size = initial_image_size // (2 ** len(encoder_hidden_dims))

        # Flattened size calculation
        self.flattened_size = encoder_hidden_dims[-1] * \
            (self.reduced_size ** 2)

        # Linear layers
        self.fc_mu = nn.Linear(self.flattened_size, dims[0])
        self.fc_logvar = nn.Linear(self.flattened_size, dims[0])

        for i in range(len(dims) - 1):
            self.add_module(
                f"fc_mu_{i}",
                nn.Linear(dims[i], dims[i + 1]),
            )
            self.add_module(
                f"fc_logvar_{i}",
                nn.Linear(dims[i], dims[i + 1]),
            )
        # initialize weights at 0s
        self.fc_logvar.weight.data.zero_()
        self.fc_logvar.bias.data.zero_()

        for i in range(len(dims) - 1):
            self.__getattr__(f"fc_logvar_{i}").weight.data.zero_()
            self.__getattr__(f"fc_logvar_{i}").bias.data.zero_()


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # Return all the means and logvars
        mus = []
        logvars = []

        # Instead of flattening, we use a global pooling layer
        x = torch.flatten(x, start_dim=1)

        # Get the mean and logvar for the first layer
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Append to the list
        mus.append(mu)
        logvars.append(logvar)

        # Get the mean and logvar for the rest of the layers
        for i in range(len(self.dims) - 1):
            # pylint: disable=unnecessary-dunder-call
            mu = self.__getattr__(f"fc_mu_{i}")(mu)
            logvar = self.__getattr__(f"fc_logvar_{i}")(logvar)

            mus.append(mu)
            logvars.append(logvar)

        # Return the means and logvars
        return mus, logvars
