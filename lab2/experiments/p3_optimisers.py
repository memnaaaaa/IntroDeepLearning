# lab2/experiments/p3_optimisers.py
# This script compares different optimizers (SGD, SGD with Nesterov momentum, and Adam) on the BaselineNet model using the FashionMNIST dataset.

import torch.nn as nn # PyTorch neural network module
import torch.optim as optim # PyTorch optimization algorithms
import matplotlib.pyplot as plt # Plotting library

from src.data       import get_loaders # Data loading function
from src.models     import BaselineNet # Baseline neural network model
from src.engine     import train_one_epoch, evaluate # Training and evaluation functions
from src.utils      import set_seed, plot_history # Utility functions
from src.constants  import DEVICE, EPOCHS, LR # Constants

set_seed() # For reproducibility
train_ld, val_ld = get_loaders(augment=False) # Get data loaders
criterion = nn.CrossEntropyLoss() # Loss function

# ------------------------------------------------------------------
# Reusable experiment runner
# ------------------------------------------------------------------
def train_with_optimiser(name, optimizer):
    print(f"\n>>> {name}") # Print current optimizer name
    hist = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc', 'time')} # Initialize history dictionary

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, epoch_time = train_one_epoch(model, train_ld, optimizer, criterion) # Train for one epoch
        val_loss, val_acc = evaluate(model, val_ld, criterion) # Evaluate on validation set

        # Record metrics
        for k, v in zip(hist, (tr_loss, val_loss, tr_acc, val_acc, epoch_time)):
            hist[k].append(v) # Append metrics to history

        # Print epoch summary
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")

    # Plot training history
    plot_history(hist, f"Problem 3 – {name}", f"outputs/p3_{name.replace(' ', '_')}.png")
    return hist

# ------------------------------------------------------------------
# 1. vanilla SGD
# ------------------------------------------------------------------
model = BaselineNet().to(DEVICE) # Instantiate and move model to device
optimizer = optim.SGD(model.parameters(), lr=LR) # SGD optimizer
sgd_hist = train_with_optimiser("SGD", optimizer) # Run training with SGD

# ------------------------------------------------------------------
# 2. SGD + Nesterov momentum
# ------------------------------------------------------------------
model = BaselineNet().to(DEVICE) # Instantiate and move model to device
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True) # SGD with Nesterov momentum
sgd_nest_hist = train_with_optimiser("SGD+Nesterov", optimizer) # Run training with SGD+Nesterov

# ------------------------------------------------------------------
# 3. Adam
# ------------------------------------------------------------------
model = BaselineNet().to(DEVICE) # Instantiate and move model to device
optimizer = optim.Adam(model.parameters(), lr=LR) # Adam optimizer
adam_hist = train_with_optimiser("Adam", optimizer) # Run training with Adam

# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------
# final validation accuracies
summary_acc = {k: hist['val_acc'][-1] for k, hist in
               zip(("SGD", "SGD+Nesterov", "Adam"), (sgd_hist, sgd_nest_hist, adam_hist))}
# average epoch times
summary_time = {k: sum(hist['time']) / EPOCHS for k, hist in
                zip(("SGD", "SGD+Nesterov", "Adam"), (sgd_hist, sgd_nest_hist, adam_hist))}

print("\nFinal validation accuracy:")
for k, v in summary_acc.items():
    print(f"  {k:<15} {v:.4f}")
print("\nAvg seconds / epoch:")
for k, v in summary_time.items():
    print(f"  {k:<15} {v:.2f}s")

# overlay convergence plot
plt.figure()
for label, hist in zip(("SGD", "SGD+Nesterov", "Adam"), (sgd_hist, sgd_nest_hist, adam_hist)):
    plt.plot(range(1, EPOCHS + 1), hist['val_acc'], label=label)
plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
plt.title("Convergence speed – optimisers")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig("outputs/p3_convergence_overlay.png")
plt.close()