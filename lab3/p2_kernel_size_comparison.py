# lab3/p2_kernel_size_comparison.py
# This script compares CNN models with different kernel sizes on the MNIST dataset and plots their training history.

# Import necessary libraries and modules
import torch                                # PyTorch library for tensor computations and deep learning
import torch.nn as nn                       # Neural network module
import torch.nn.functional as F             # Functional API for neural networks
import torch.optim as optim                 # Optimization algorithms
import torchvision.transforms as transforms # Image transformations
import torchvision.datasets as datasets     # Standard datasets
from torch.utils.data import DataLoader     # Data loading utility
import time                                 # Time measurement
import matplotlib.pyplot as plt             # Plotting library

# Define transformations and load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create a function to generate CNN models with different kernel sizes
class CNN(nn.Module):
    def __init__(self, kernel_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=kernel_size//2) # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=kernel_size//2) # Second convolutional layer
        self.fc1 = nn.Linear(64*7*7, 10) # Fully connected layer

    # Forward pass
    def forward(self, x):
        x = torch.relu(self.conv1(x)) # Apply ReLU activation
        x = F.max_pool2d(x, 2) # Max pooling
        x = torch.relu(self.conv2(x)) # Apply ReLU activation
        x = F.max_pool2d(x, 2) # Max pooling
        x = x.view(-1, 64*7*7) # Flatten the tensor
        x = self.fc1(x) # Fully connected layer
        return x # Output logits

# Function to get model with specified kernel size
def get_model(kernel_size, device):
    return CNN(kernel_size).to(device)

# Set up training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device configuration
criterion = nn.CrossEntropyLoss() # Loss function

# Train models with different kernel sizes and record metrics
kernel_sizes = [3, 5, 7]
results = [] # List to store results

# Collect per-epoch train/test accuracies for plotting
train_accuracies = []
test_accuracies = []
epochs = 10

for kernel_size in kernel_sizes:
    model = get_model(kernel_size, device) # Instantiate the CNN model with specific kernel size
    optimizer = optim.SGD(model.parameters(), lr=0.01) # Optimizer
    start_time = time.time() # Start time measurement

    # Lists to store per-epoch accuracies
    per_epoch_train = []
    per_epoch_test = []

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        running_correct = 0
        running_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # Move data to device
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            _, preds = torch.max(outputs, 1) # Get predictions
            running_total += labels.size(0) # Update total count
            running_correct += (preds == labels).sum().item() # Update correct count

        train_acc = 100.0 * running_correct / running_total # Calculate training accuracy
        per_epoch_train.append(train_acc) # Record training accuracy

        # Evaluate on test set each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device) # Move data to device
                outputs = model(images) # Forward pass
                _, predicted = torch.max(outputs.data, 1) # Get predictions
                total += labels.size(0) # Update total count
                correct += (predicted == labels).sum().item() # Update correct count
        test_acc = 100.0 * correct / total # Calculate test accuracy
        per_epoch_test.append(test_acc) # Record test accuracy

        print(f'Kernel {kernel_size} Epoch {epoch:02d}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

    # Record final metrics
    end_time = time.time() # End time measurement
    model_size = sum(p.numel() for p in model.parameters()) # Model size in parameters
    final_test_acc = per_epoch_test[-1] # Final test accuracy
    results.append((kernel_size, final_test_acc, end_time - start_time, model_size)) # Append results
    train_accuracies.append(per_epoch_train) # Append per-epoch train accuracies
    test_accuracies.append(per_epoch_test) # Append per-epoch test accuracies

print("Kernel Size, Accuracy, Training Time (s), Model Size (params)")
for result in results:
    print(f"{result[0]}x{result[0]}, {result[1]:.2f}%, {result[2]:.2f}s, {result[3]/1e6:.2f}M")

# Plot training and test accuracy for different kernel sizes
plt.figure(figsize=(10, 6))
for i, kernel_size in enumerate(kernel_sizes):
    plt.plot(train_accuracies[i], label=f'Train Acc - {kernel_size}x{kernel_size}') # Plot training accuracy
    plt.plot(test_accuracies[i], '--', label=f'Test Acc - {kernel_size}x{kernel_size}') # Plot test accuracy
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy for Different Kernel Sizes')
plt.legend()
plt.tight_layout()
plt.show() # Display the plot
