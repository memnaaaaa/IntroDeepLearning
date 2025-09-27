# lab2/src/engine.py
# This file contains functions to train and evaluate a PyTorch model for one epoch, including optional L1 regularization.

import torch, time # PyTorch library and time module for performance measurement
from .constants import DEVICE # Import device constant
import time

# Function to train the model for one epoch
def train_one_epoch(model, loader, optimiser, criterion, l1_lambda=0.0):
    model.train() # Set model to training mode
    running_loss, correct, total = 0.0, 0, 0 # Initialize metrics
    start = time.time() # Start time for epoch

    for x, y in loader: # Iterate over batches
        x, y = x.to(DEVICE), y.to(DEVICE) # Move data to the specified device
        optimiser.zero_grad(set_to_none=True) # Zero the gradients
        out = model(x) # Forward pass
        loss = criterion(out, y) # Compute loss
        if l1_lambda: # Apply L1 regularization if lambda is non-zero
            loss += l1_lambda * sum(p.abs().sum() for p in model.parameters()) # L1 regularization
        loss.backward() # Backward pass
        optimiser.step() # Update weights
        running_loss += loss.item() # Accumulate loss
        _, pred = out.max(1) # Get predicted classes
        total += y.size(0) # Update total count
        correct += pred.eq(y).sum().item() # Update correct predictions

    epoch_time = time.time() - start # Calculate epoch duration
    return running_loss / len(loader), correct / total, epoch_time # Return average loss, accuracy, and epoch time

# Function to evaluate the model on a validation set
@torch.no_grad() # Disable gradient computation for evaluation
def evaluate(model, loader, criterion):
    model.eval() # Set model to evaluation mode
    running_loss, correct, total = 0.0, 0, 0 # Initialize metrics
    for x, y in loader: # Iterate over batches
        x, y = x.to(DEVICE), y.to(DEVICE) # Move data to the specified device
        out = model(x) # Forward pass
        running_loss += criterion(out, y).item()
        _, pred = out.max(1) # Get predicted classes
        total += y.size(0) # Update total count
        correct += pred.eq(y).sum().item() # Update correct predictions
    return running_loss / len(loader), correct / total # Return average loss and accuracy