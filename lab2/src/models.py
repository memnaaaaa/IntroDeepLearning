# lab2/src/models.py
# This file defines a baseline neural network model for classifying FashionMNIST images using PyTorch.

import torch.nn as nn # PyTorch neural network module

# Baseline neural network model for FashionMNIST classification
class BaselineNet(nn.Module):
    def __init__(self, dropout=0.0): # Initialize with optional dropout
        super().__init__() # Call the parent class constructor
        # Define the network architecture
        layers = [nn.Flatten(), # Flatten the 28x28 images to 784-dimensional vectors
                  nn.Linear(28*28, 256), # First hidden layer with 256 neurons
                  nn.ReLU(), # ReLU activation
                  nn.Linear(256, 128), # Second hidden layer with 128 neurons
                  nn.ReLU()] # ReLU activation
        if dropout > 0: # Add dropout layer if specified
            layers += [nn.Dropout(dropout)] # Dropout for regularization
        layers += [nn.Linear(128, 10)] # Output layer for 10 classes
        self.net = nn.Sequential(*layers) # Combine layers into a sequential model

    # Forward pass
    def forward(self, x):
        return self.net(x) # Pass input through the network