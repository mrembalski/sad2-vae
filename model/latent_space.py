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
            dims: list[int],
            flattened_size: int,
        ):
        super().__init__()
        print("LatentSpace:")

        self.dims = dims
        self.flattened_size = flattened_size

        self.fc_mu = nn.Linear(self.flattened_size, dims[0])
        self.fc_logvar = nn.Linear(self.flattened_size, dims[0])
        print(f"\t{self.flattened_size} -> {dims[0]}")

        # Additional layers for hierarchical structure
        self.additional_fc_mu = nn.ModuleList()
        self.additional_fc_logvar = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.additional_fc_mu.append(nn.Linear(dims[i], dims[i + 1]))
            self.additional_fc_logvar.append(nn.Linear(dims[i], dims[i + 1]))
            print(f"\t{dims[i]} -> {dims[i + 1]}")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # Flatten the input for the linear layers
        x = torch.flatten(x, start_dim=1)

        # First layer mean and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Lists to store the hierarchical means and logvars
        mus = [mu]
        logvars = [logvar]

        # Process through additional layers
        for i in range(len(self.dims) - 1):
            mu = self.additional_fc_mu[i](mu)
            logvar = self.additional_fc_logvar[i](logvar)

            mus.append(mu)
            logvars.append(logvar)

        # Return the hierarchical means and logvars
        return mus, logvars
