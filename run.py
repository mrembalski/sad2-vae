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

base_transform = [
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[1]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
]

train_transform = transforms.Compose(
    base_transform + [
        transforms.Lambda(clahe),
        transforms.RandomRotation(degrees=90),
])

vae_model = HVAE(
    initial_image_size = 256,
    input_channels = 1,
    output_channels = 1,
    # Number of passes:
    encoder_hidden_dims = [32, 64, 128, 256],
    latent_dims = [256],
    learning_rate = 5e-4,
    # We can afford to have a high beta because of the annealing
    beta = 1.0,
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
        KLAnnealingCallback(max_kl_coefficient=1.0, anneal_steps=len(train_dataloader)),
        LearningRateMonitor(logging_interval='step'),
    ],
    logger=logger,
)

val_dataset = datasets.ImageFolder(root='SMDG-19/val', transform=train_transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

trainer.fit(vae_model, train_dataloader, val_dataloader)

# Save checkpoint
torch.save(vae_model.state_dict(), "model.pt")
vae_model.load_state_dict(torch.load("model.pt"))