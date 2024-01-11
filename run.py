# pylint: disable=missing-module-docstring, missing-function-docstring
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import datasets, transforms # type: ignore
from skimage import exposure
import torch
from torch import nn

from model.hvae import HVAE, KLAnnealingCallback

def clahe(x: torch.Tensor):
    x = x.numpy()
    x = exposure.equalize_adapthist(x)
    x = torch.from_numpy(x)
    return x

IS_GREYSCALE = True
INITIAL_IMAGE_SIZE = 256

base_transform = [
    transforms.Resize((INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE)),
    transforms.ToTensor(),
    # Get green channel if greyscale else turn to RGB
    transforms.Lambda(lambda x: x[1].unsqueeze(0) if IS_GREYSCALE else x),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
]

train_transform = transforms.Compose(
    base_transform + [
        transforms.Lambda(clahe),
])

val_transform = transforms.Compose(
    base_transform,
)

vae_model = HVAE(
    initial_image_size = INITIAL_IMAGE_SIZE,
    input_channels = 1 if IS_GREYSCALE else 3,
    output_channels = 1 if IS_GREYSCALE else 3,
    encoder_hidden_dims = [64, 128, 256, 512, 1024],
    latent_dims = [512],
    learning_rate = 1e-3,
    # Raczej między (0, 1) + cyclic annealing
    beta = 1 / 8,
    stride = 2,
    expansion = 2,
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

vae_model.latent_space.apply(init_weights)

logger = TensorBoardLogger(
    "tb_logs", 
    name="HVAE",
)

train_dataset = datasets.ImageFolder(root='SMDG-19/train', transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

trainer = L.Trainer(
    max_epochs=10,
    callbacks=[
        ModelSummary(max_depth=4),
        KLAnnealingCallback(anneal_steps=len(train_dataloader)),
        LearningRateMonitor(logging_interval='step'),
    ],
    log_every_n_steps=10,
    logger=logger,
)

val_dataset = datasets.ImageFolder(root='SMDG-19/val', transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

trainer.fit(vae_model, train_dataloader, val_dataloader)

# Save checkpoint
torch.save(vae_model.state_dict(), "model.pt")
vae_model.load_state_dict(torch.load("model.pt"))
