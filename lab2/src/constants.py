# lab2/src/constants.py
# This file contains constant values used throughout the lab2 project, such as configuration parameters, file paths, and other fixed values to ensure consistency and easy maintenance.

import torch # PyTorch library for tensor computations and deep learning
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device to GPU if available, else CPU
BATCH_SIZE = 64 # Batch size for data loaders
EPOCHS     = 10  # Number of training epochs
LR         = 0.01 # Learning rate