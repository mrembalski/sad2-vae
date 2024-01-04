# pylint: disable=missing-module-docstring
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import datasets, transforms # type: ignore

from model.hvae import HVAE

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

vae_model = HVAE(
    initial_image_size = 224,
    input_channels = 3,
    output_channels = 3,
    encoder_hidden_dims = [32, 64, 128, 256, 512],
    latent_dims = [256],
    learning_rate = 1e-4,
    beta = 1.0,
)

logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = L.Trainer(
    max_epochs=3,
    callbacks=[
        ModelSummary(max_depth=2),
    ],
    logger=logger,
)

train_dataset = datasets.ImageFolder(root='SMDG-19/train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
trainer.fit(vae_model, train_dataloader)

test_dataset = datasets.ImageFolder(root='SMDG-19/test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
trainer.test(vae_model, test_dataloader)
