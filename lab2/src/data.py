# lab2/src/data.py
# This file contains functions to load and preprocess the FashionMNIST dataset using PyTorch's torchvision library.

import torchvision.transforms as T, torchvision.datasets as datasets # Datasets and transformations
from torch.utils.data import DataLoader # Data loading utilities
from .constants import BATCH_SIZE # Import batch size constant

stats = ((0.5,), (0.5,)) # Mean and std for normalization

# Function to get data loaders for training and validation sets
def get_loaders(augment=False):
    base = [T.ToTensor(), T.Normalize(*stats)] # Base transformations: convert to tensor and normalize
    # Add data augmentation if specified (only for training set)
    train_tf = T.Compose(([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10)
    ] if augment else []) + base)
    # Validation transformations (no augmentation)
    val_tf   = T.Compose(base)

    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_tf) # Training dataset with optional augmentation
    val_set   = datasets.FashionMNIST(root='./data', train=False, download=True, transform=val_tf) # Validation dataset without augmentation
    train_ld = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # Shuffle training data
    val_ld   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False) # No shuffle for validation data
    return train_ld, val_ld # Return the data loaders