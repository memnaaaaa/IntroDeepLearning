# lab2/src/schedulers.py
# This file provides a function to create different learning rate schedulers for a given optimizer in PyTorch.

import torch.optim.lr_scheduler as LS # PyTorch learning rate schedulers

# Function to get a learning rate scheduler by name
def get_scheduler(name, optimizer):
    if name == 'Step':
        return LS.StepLR(optimizer, step_size=3, gamma=0.5)      # ÷2 every 3 epochs
    if name == 'Exponential':
        return LS.ExponentialLR(optimizer, gamma=0.9)            # ×0.9 every epoch
    if name == 'Cosine':
        return LS.CosineAnnealingLR(optimizer, T_max=10)         # one cosine cycle
    if name == 'Plateau':
        return LS.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                    patience=2, verbose=False)
    raise ValueError(name) # Unknown scheduler name