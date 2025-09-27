# lab2/experiments/p2_regularisers.py
# This script trains neural networks with various regularization techniques on the FashionMNIST dataset and plots the training histories.

import torch.nn as nn, torch.optim as optim # PyTorch libraries
import matplotlib.pyplot as plt # Plotting library

from src.data       import get_loaders # Data loading function
from src.models     import BaselineNet # Baseline neural network model
from src.engine     import train_one_epoch, evaluate # Training and evaluation functions
from src.utils      import set_seed, plot_history # Utility functions
from src.constants  import DEVICE, EPOCHS, LR # Constants

set_seed() # Set random seed for reproducibility
train_ld, val_ld = get_loaders(augment=False) # Get data loaders without augmentation
crit = nn.CrossEntropyLoss() # Cross-entropy loss

# Reusable experiment runner with specified model, optimizer, and L1 regularization
def run_experiment(name, model, opt, l1_lambda=0.0):
    hist = {k: [] for k in ('train_loss','val_loss','train_acc','val_acc')} # Initialize history dictionary
    # Training loop
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, opt, crit, l1_lambda) # Train for one epoch
        val_loss, val_acc = evaluate(model, val_ld, crit) # Evaluate on validation set
        # Record metrics
        for k, v in zip(hist, (tr_loss, val_loss, tr_acc, val_acc)):
            hist[k].append(v) # Append metrics to history
        # Print epoch summary
        print(f'Epoch {epoch:02d} | '
              f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    # Plot training history
    plot_history(hist, f'Problem 2 – {name}', f'outputs/p2_{name.replace(" ","_")}.png') # Save plot with experiment name
    return hist['val_acc'][-1] # Return final validation accuracy

# -----------------------------------------------------------
# 1. L2 weight decay
# -----------------------------------------------------------
print('\n>>> L2 weight-decay 1e-4')
net = BaselineNet().to(DEVICE) # Instantiate and move model to device
opt = optim.SGD(net.parameters(), lr=LR, weight_decay=1e-4) # SGD optimizer with L2 weight decay
l2_acc = run_experiment("L2 weight-decay 1e-4", net, opt) # Run experiment with L2 regularization

# -----------------------------------------------------------
# 2. L1 regularization 
# -----------------------------------------------------------
print('\n>>> L1 lambda 1e-5')
net = BaselineNet().to(DEVICE) # Instantiate and move model to device
opt = optim.SGD(net.parameters(), lr=LR) # SGD optimizer without weight decay
l1_acc = run_experiment("L1 lambda 1e-5", net, opt, l1_lambda=1e-5) # Run experiment with L1 regularization

# -----------------------------------------------------------
# 3. Dropout
# -----------------------------------------------------------
dropout_accs = {} # Dictionary to store accuracies for different dropout rates
# Experiment with different dropout probabilities
for p in (0.2, 0.5):
    print(f'\n>>> Dropout p={p}') # Print current dropout probability
    net = BaselineNet(dropout=p).to(DEVICE) # Instantiate model with dropout and move to device
    opt = optim.SGD(net.parameters(), lr=LR) # SGD optimizer without weight decay
    dropout_accs[f'Dropout p={p}'] = run_experiment(f'Dropout p={p}', net, opt) # Run experiment and store accuracy

# -----------------------------------------------------------
# 4. Data augmentation
# -----------------------------------------------------------
print('\n>>> Data Augmentation')
train_aug, val_aug = get_loaders(augment=True) # Get data loaders with augmentation
net = BaselineNet().to(DEVICE) # Instantiate and move model to device
opt = optim.SGD(net.parameters(), lr=LR) # SGD optimizer without weight decay
hist_aug = {k: [] for k in ('train_loss','val_loss','train_acc','val_acc')} # Initialize history dictionary
# Training loop with data augmentation
for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc = train_one_epoch(net, train_aug, opt, crit) # Train for one epoch
    val_loss, val_acc = evaluate(net, val_aug, crit) # Evaluate on validation set
    # Record metrics
    for k, v in zip(hist_aug, (tr_loss, val_loss, tr_acc, val_acc)):
        hist_aug[k].append(v) # Append metrics to history
    # Print epoch summary
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
# Plot training history for data augmentation
plot_history(hist_aug, 'Problem 2 – Data Augmentation', 'outputs/p2_DataAug.png')
aug_acc = hist_aug['val_acc'][-1]

# -----------------------------------------------------------
# summary of results
# -----------------------------------------------------------
summary = {
    'L2 weight-decay 1e-4': l2_acc,
    'L1 lambda 1e-5':       l1_acc,
    **dropout_accs,
    'Data Augmentation':    aug_acc
}
print('\nFinal validation accuracies:')
# Print final validation accuracies for all experiments
for k, v in summary.items():
    print(f'  {k:<20} {v:.4f}')

# Bar plot of final validation accuracies
plt.bar(summary.keys(), summary.values()) # Create bar plot
plt.ylabel('Final val accuracy') # Label y-axis
plt.xticks(rotation=25, ha='right') # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent overlap
plt.savefig('outputs/p2_summary.png') # Save the plot
plt.close() # Close the plot to free memory
