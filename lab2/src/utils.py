# lab2/src/utils.py
# This file contains utility functions for setting random seeds and plotting training history.

import matplotlib.pyplot as plt, numpy as np, torch # Plotting, numerical operations, and PyTorch library

# Function to set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed) # Set seed for CPU
    if torch.cuda.is_available(): # If CUDA is available, set seed for GPU
        torch.cuda.manual_seed(seed)

# Function to plot training and validation loss and accuracy
def plot_history(hist, title, savepath):
    fig, ax = plt.subplots(1, 2, figsize=(11,3)) # Create subplots for loss and accuracy
    epochs = np.arange(1, len(hist['train_loss'])+1) # Epoch numbers
    ax[0].plot(epochs, hist['train_loss'], label='train') # Plot training loss
    ax[0].plot(epochs, hist['val_loss'],   label='val') # Plot validation loss
    ax[0].set_title(f'{title} – Loss'); ax[0].legend(); ax[0].grid()
    ax[1].plot(epochs, hist['train_acc'], label='train') # Plot training accuracy
    ax[1].plot(epochs, hist['val_acc'],   label='val') # Plot validation accuracy
    ax[1].set_title(f'{title} – Acc'); ax[1].legend(); ax[1].grid() # Add grid for better readability
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(savepath) # Save the plot to the specified path
    plt.close() # Close the plot to free memory
    