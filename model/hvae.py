# pylint: disable=missing-function-docstring, missing-module-docstring, line-too-long, missing-class-docstring, import-error, no-name-in-module
from PIL import Image # type: ignore

import numpy as np

import torch
import torch.nn.functional as F
import lightning as L
from typeguard import typechecked

from model.encoder import Encoder
from model.latent_space import LatentSpace
from model.decoder import Decoder

def initialize_weights_he(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class HVAE(L.LightningModule):
    @typechecked
    # pylint: disable=too-many-arguments
    def __init__(self,
        initial_image_size: int,
        input_channels: int,
        output_channels: int,
        encoder_hidden_dims: list[int],
        latent_dims: list[int],
        learning_rate: float,
        beta: float,
    ):
        super().__init__()
        self.beta = beta
        self.encoder = Encoder(input_channels, encoder_hidden_dims)
        self.latent_space = LatentSpace(initial_image_size, encoder_hidden_dims, latent_dims)
        self.decoder = Decoder(initial_image_size, output_channels, encoder_hidden_dims, latent_dims[-1])
        self.learning_rate = learning_rate

    def _step(self, x):
        x = self.encoder(x)
        mus, logvars = self.latent_space(x)
        last_z = self.latent_space.reparameterize(mus[-1], logvars[-1])
        reconstructed_x = self.decoder(last_z)
        return reconstructed_x, mus, logvars

    def _log_first_image(self, x, reconstructed_x):
        _x = x[0].detach().cpu().numpy()
        _rx = reconstructed_x[0].detach().cpu().numpy()

        if np.isnan(_x).any() or np.isnan(_rx).any():
            print("NaN encountered")

        _x = (_x * 255).astype('uint8')
        _rx = (_rx * 255).astype('uint8')

        Image.fromarray(_x.transpose(1, 2, 0)).save("training_tensors/x.png")
        Image.fromarray(_rx.transpose(1, 2, 0)).save("training_tensors/rx.png")

    # pylint: disable=arguments-differ
    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed_x, mus, logvars = self._step(x)
        loss = self.compute_loss(x, reconstructed_x, mus, logvars)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 10 == 0:
            self._log_first_image(x, reconstructed_x)

        return loss

    # pylint: disable=arguments-differ
    def validation_step(self, batch, _):
        x, _ = batch
        reconstructed_x, mus, logvars = self._step(x)
        loss = self.compute_loss(x, reconstructed_x, mus, logvars)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / 100.0),
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def kl_divergence_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def compute_loss(self, x, reconstructed_x, mus, logvars):
        bce_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='mean')

        kl_loss = torch.tensor(0.0, device=self.device)
        for mu, logvar in zip(mus, logvars):
            kl_loss += self.kl_divergence_loss(mu, logvar)

        bkl_loss = self.beta * kl_loss

        print(f"bce_loss: {bce_loss:.3f}")
        print(f"bkl_loss: {bkl_loss:.3f}")

        loss = bce_loss + bkl_loss
        return loss
