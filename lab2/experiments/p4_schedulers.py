# lab2/experiments/p4_schedulers.py
# This script compares different learning rate schedulers (Step, Exponential, Cosine, and Plateau) on the BaselineNet model using the FashionMNIST dataset.

import torch.nn as nn, torch.optim as optim # PyTorch libraries
import matplotlib.pyplot as plt # Plotting library

from src.data       import get_loaders # Data loading function
from src.models     import BaselineNet # Baseline neural network model
from src.engine     import train_one_epoch, evaluate # Training and evaluation functions
from src.utils      import set_seed, plot_history # Utility functions
from src.schedulers import get_scheduler # Function to get learning rate schedulers
from src.constants  import DEVICE, EPOCHS # Constants

set_seed() # For reproducibility
train_ld, val_ld = get_loaders(augment=False) # Get data loaders
criterion = nn.CrossEntropyLoss() # Loss function

# ------------------------------------------------------------------
# Reusable function to train with a given scheduler
# ------------------------------------------------------------------
def train_with_scheduler(name, scheduler):
    print(f"\n>>> Scheduler: {name}") # Print current scheduler name
    # one model / optimiser for this run
    model    = BaselineNet().to(DEVICE) # Instantiate and move model to device
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True) # SGD with Nesterov momentum
    scheduler = get_scheduler(name, optimizer) # Get the specified scheduler

    hist = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr')} # Initialize history dictionary
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _ = train_one_epoch(model, train_ld, optimizer, criterion) # Train for one epoch
        val_loss, val_acc  = evaluate(model, val_ld, criterion) # Evaluate on validation set

        # Record metrics
        for k, v in zip(hist, (tr_loss, val_loss, tr_acc, val_acc,
                               optimizer.param_groups[0]['lr'])):
            hist[k].append(v) # Append metrics to history

        # Print epoch summary
        print(f"Ep {epoch:02d} | tr-loss {tr_loss:.4f} tr-acc {tr_acc:.4f} | "
              f"val-acc {val_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}")

        # Step scheduler
        if name == 'Plateau':
            scheduler.step(val_acc)          # needs metric
        else:
            scheduler.step()                 # epoch-based

    # Plot training history
    plot_history(hist, f"Problem 4 – {name}", f"outputs/p4_{name}.png")
    return hist

# ------------------------------------------------------------------
# run the four schedulers
# ------------------------------------------------------------------
scheds = ('Step', 'Exponential', 'Cosine', 'Plateau') # Scheduler names to compare
histories = {s: train_with_scheduler(s, None) for s in scheds} # Train with each scheduler and store histories

# ------------------------------------------------------------------
# summary & overlay plot
# ------------------------------------------------------------------
print("\nFinal validation accuracy:")
for s in scheds:
    print(f"  {s:<12} {histories[s]['val_acc'][-1]:.4f}")

# Overlay plot of validation accuracy
plt.figure()
for s in scheds:
    plt.plot(range(1, EPOCHS+1), histories[s]['val_acc'], label=s)
plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
plt.title("LR schedulers – convergence speed")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig("outputs/p4_scheduler_overlay.png")
plt.close()
