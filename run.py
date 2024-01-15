# pylint: disable=missing-module-docstring, missing-function-docstring
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import datasets, transforms  # type: ignore
from skimage import exposure
import torch
from torch import nn

from model.hvae import HVAE, KLAnnealingCallback

torch.set_float32_matmul_precision('medium')


def clahe(x: torch.Tensor):
    x = x.numpy()
    x = exposure.equalize_adapthist(x)
    x = torch.from_numpy(x)
    return x


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


IS_GREYSCALE = True
INITIAL_IMAGE_SIZE = 128

base_transform = [
    transforms.Resize((INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE)),
    # Turn to RGB
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: \
        (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0) if IS_GREYSCALE else x),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
]

train_transform = transforms.Compose(
    base_transform + [
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Lambda(clahe),
])

val_transform = transforms.Compose(
    base_transform + [
    transforms.Lambda(clahe),
])

train_dataset = datasets.ImageFolder(root='SMDG-19/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='SMDG-19/val', transform=val_transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64,
    num_workers=40, persistent_workers=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=64,
    num_workers=40, persistent_workers=True)


vae_model = HVAE(
    initial_image_size=INITIAL_IMAGE_SIZE,
    input_channels=1 if IS_GREYSCALE else 3,
    output_channels=1 if IS_GREYSCALE else 3,
    # Remember, that first layer has stride=1.
    # It acutally helped to deblurr the images a lot.
    encoder_hidden_dims=[32] + [64, 128],
    latent_dims=[256],
    learning_rate=1e-4,
    # Remember, that effectively beta is halved
    # due to annealing.
    beta=2,
    stride=2,
)

vae_model.apply(initialize_weights)

logger = TensorBoardLogger(
    "tb_logs",
    name="HVAE",
    version="latent_dims=[256_new]",
)

trainer = L.Trainer(
    max_epochs=50,
    callbacks=[
        ModelSummary(max_depth=5),
        KLAnnealingCallback(anneal_steps=(len(train_dataloader) // 2) * 5),
        LearningRateMonitor(logging_interval='step'),
    ],
    log_every_n_steps=1,
    logger=logger,
    strategy="ddp",
    devices=2,
)

trainer.fit(vae_model, train_dataloader)
