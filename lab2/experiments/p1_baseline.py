# lab2/experiments/p1_baseline.py
# This script trains a baseline neural network on the FashionMNIST dataset and plots the training history.

import torch.nn as nn, torch.optim as optim # PyTorch libraries

from src.data       import get_loaders # Data loading function
from src.models     import BaselineNet # Baseline neural network model
from src.engine     import train_one_epoch, evaluate # Training and evaluation functions
from src.utils      import set_seed, plot_history # Utility functions
from src.constants  import DEVICE, EPOCHS, LR # Constants

set_seed() # Set random seed for reproducibility
train_ld, val_ld = get_loaders(augment=False) # Get data loaders without augmentation
model = BaselineNet().to(DEVICE) # Instantiate and move model to device
opt   = optim.SGD(model.parameters(), lr=LR) # SGD optimizer
crit  = nn.CrossEntropyLoss() # Cross-entropy loss

hist = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]} # Initialize history dictionary
# Training loop
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_ld, opt, crit) # Train for one epoch
    val_loss, val_acc = evaluate(model, val_ld, crit) # Evaluate on validation set

    # Record metrics
    hist['train_loss'].append(tr_loss)
    hist['val_loss'].append(val_loss)
    hist['train_acc'].append(tr_acc)
    hist['val_acc'].append(val_acc)

    # Print epoch summary
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

# Plot training history
plot_history(hist, 'Problem 1 – Baseline', 'outputs/p1_baseline.png')
