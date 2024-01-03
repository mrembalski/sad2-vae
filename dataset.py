"""
This file contains the class to load the SMDG dataset.
"""
import os

import torch
from torch.utils.data import TensorDataset

from constants import NORMALIZED_DATASET

class SMDGDataset(TensorDataset): # pylint: disable=too-few-public-methods
    """
    Simple class to load the SMDG dataset from the normalized tensors.
    """
    def __init__(self, split: str = 'train'):

        # Check if stacked tensors already exist
        if os.path.exists(f"{NORMALIZED_DATASET}/{split}/stacked_tensors.pt"):
            tensors, classes = torch.load(f"{NORMALIZED_DATASET}/{split}/stacked_tensors.pt")
        else:
            tensors = []
            classes = []

            for tensor_filename in os.listdir(f"{NORMALIZED_DATASET}/{split}/0/"):
                tensors.append(torch.load(f"{NORMALIZED_DATASET}/{split}/0/{tensor_filename}"))
                classes.append(torch.tensor(0))

            for tensor_filename in os.listdir(f"{NORMALIZED_DATASET}/{split}/1/"):
                tensors.append(torch.load(f"{NORMALIZED_DATASET}/{split}/1/{tensor_filename}"))
                classes.append(torch.tensor(1))

            if tensors[0].ndim == 2:
                # If the tensors are 2D, add a channel dimension
                tensors = [tensor.unsqueeze(0) for tensor in tensors]

            tensors = torch.stack(tensors)
            classes = torch.stack(classes)

            torch.save(
                (
                    tensors,
                    classes,
                ),
                f"{NORMALIZED_DATASET}/{split}/stacked_tensors.pt"
            )

        self.tensors = (
            tensors,
            classes
        )

        print(f"Loaded {len(self.tensors[0])} tensors from {split} split.")
