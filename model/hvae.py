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

LR_ANNEALLING_STEPS = 100

class KLAnnealingCallback(L.Callback):
    def __init__(self, anneal_steps: int):
        self.anneal_steps = anneal_steps

    # pylint: disable=too-many-arguments
    def on_train_batch_start(self, trainer, pl_module, outputs, batch):
        current_step = trainer.global_step % self.anneal_steps

        new_kl_coefficient = 1 - (0.5 * (1 + np.cos(np.pi * current_step / self.anneal_steps)))
        pl_module.kl_coefficient = new_kl_coefficient

        trainer.logger.log_metrics({'kl_coeff': new_kl_coefficient}, step=trainer.global_step)
        trainer.logger.log_metrics({'kl_beta': pl_module.beta * pl_module.kl_coefficient}, step=trainer.global_step)


class HVAE(L.LightningModule):
    @typechecked
    # pylint: disable=too-many-arguments
    def __init__(self,
        initial_image_size: int,
        input_channels: int,
        output_channels: int,
        stride: int,
        encoder_hidden_dims: list[int],
        latent_dims: list[int],
        learning_rate: float,
        beta: float,
    ):
        super().__init__()
        self.beta = beta
        self.kl_coefficient = 1.0

        self.reduced_size = initial_image_size // (stride ** (len(encoder_hidden_dims) - 1))
        self.flattened_size = encoder_hidden_dims[-1] * self.reduced_size * self.reduced_size

        self.encoder = Encoder(
            input_channels=input_channels,
            stride=stride,
            encoder_hidden_dims=encoder_hidden_dims,
        )
        self.latent_space = LatentSpace(
            dims=latent_dims,
            flattened_size=self.flattened_size,
        )
        self.decoder = Decoder(
            stride=stride,
            output_channels=output_channels,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_space_dim=latent_dims[-1],
            flattened_size=self.flattened_size,
            reduced_size=self.reduced_size,
        )
        self.learning_rate = learning_rate

        width = initial_image_size
        height = initial_image_size

        weights = torch.zeros((width, height), device=self.device)
        circle_center = (width // 2, height // 2)
        circle_radius = width // 2

        for i in range(width):
            for j in range(height):
                if (i - circle_center[0]) ** 2 + (j - circle_center[1]) ** 2 <= (circle_radius + 2) ** 2:
                    weights[i, j] = 1.0
                else:
                    weights[i, j] = 0.5

        print(f"weights_sum / before_weights_sum: {torch.sum(weights) / (width * height)}")

        self.register_buffer('weights', weights)
        self.log_image(weights.unsqueeze(0), "weights")

        self.save_hyperparameters()

    def _step(self, x):
        x = self.encoder(x)
        mus, logvars = self.latent_space(x)
        last_z = self.latent_space.reparameterize(mus[-1], logvars[-1])
        reconstructed_x = self.decoder(last_z)
        return reconstructed_x, mus, logvars

    def log_image(self, x, name: str):
        assert x.ndim == 3

        x = x.detach().cpu().numpy()
        x = (x * 255).astype('uint8')
        if x.shape[0] == 3:
            Image.fromarray(x.transpose(1, 2, 0), mode='RGB').save(f"training_tensors/{name}.png")
        else:
            Image.fromarray(x[0], mode='L').save(f"training_tensors/{name}.png")

    # pylint: disable=arguments-differ
    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed_x, mus, logvars = self._step(x)
        loss = self.compute_loss(x, reconstructed_x, mus, logvars)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 5 == 0:
            self.log_image(x[0], "x")
            self.log_image(reconstructed_x[0], "rx")

        return loss

    # pylint: disable=arguments-differ
    def validation_step(self, batch, _):
        x, _ = batch
        reconstructed_x, mus, logvars = self._step(x)
        loss = self.compute_loss(x, reconstructed_x, mus, logvars)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, (step + 1) / float(LR_ANNEALLING_STEPS)),
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
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def compute_loss(self, x, reconstructed_x, mus, logvars):
        rec_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum', weight=self.weights)

        kl_loss = torch.tensor(0.0, device=self.device)
        for mu, logvar in zip(mus, logvars):
            kl_loss += self.kl_divergence_loss(mu, logvar)

        bkl_loss = self.kl_coefficient * self.beta * kl_loss

        self.log('recon_loss', rec_loss, on_step=True, on_epoch=True, logger=True)
        self.log('bkl*c_loss', bkl_loss, on_step=True, on_epoch=True, logger=True)

        loss = rec_loss + bkl_loss
        return loss
