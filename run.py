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
    x = torch.from_numpy(x).unsqueeze(0)
    return x

IS_GREYSCALE = False

base_transform = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Get green channel if greyscale else turn to RGB
    transforms.Lambda(lambda x: x[1].unsqueeze(0) if IS_GREYSCALE else x),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
]

train_transform = transforms.Compose(
    base_transform + [
        transforms.RandomRotation(degrees=90),
])

val_transform = transforms.Compose(
    base_transform,
)

vae_model = HVAE(
    initial_image_size = 224,
    input_channels = 1 if IS_GREYSCALE else 3,
    output_channels = 1 if IS_GREYSCALE else 3,
    encoder_hidden_dims = [32, 64, 128, 256],
    latent_dims = [256],
    learning_rate = 5e-4,
    # We can afford to have a high beta because of the annealing
    beta = 128.0,
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


vae_model.apply(init_weights)

logger = TensorBoardLogger(
    "tb_logs", 
    name="HVAE",
)

train_dataset = datasets.ImageFolder(root='SMDG-19/train', transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

trainer = L.Trainer(
    max_epochs=10,
    callbacks=[
        ModelSummary(max_depth=3),
        KLAnnealingCallback(anneal_steps=len(train_dataloader)),
        LearningRateMonitor(logging_interval='step'),
    ],
    logger=logger,
)

val_dataset = datasets.ImageFolder(root='SMDG-19/val', transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

trainer.fit(vae_model, train_dataloader, val_dataloader)

# Save checkpoint
torch.save(vae_model.state_dict(), "model.pt")
vae_model.load_state_dict(torch.load("model.pt"))
