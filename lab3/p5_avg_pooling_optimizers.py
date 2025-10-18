# lab3/p5_avg_pooling_optimizers.py
# This script implements a CNN with average pooling and compares different optimizers (SGD, RMSprop, Adam) on the MNIST dataset.

# Import necessary libraries and modules
import torch                                # PyTorch library for tensor computations and deep learning
import torch.nn as nn                       # Neural network module from PyTorch
import torch.optim as optim                 # Optimization algorithms from PyTorch
import torchvision.transforms as transforms # Image transformations from torchvision
import torchvision.datasets as datasets     # Datasets from torchvision
from torch.utils.data import DataLoader     # DataLoader for batching and loading data
import matplotlib.pyplot as plt             # For plotting graphs

# Define transformations and load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create a CNN model with average pooling
class CNN(nn.Module):
    def __init__(self): # Initialize the CNN model
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Second convolutional layer
        self.avgpool = nn.AvgPool2d(2) # Average pooling layer
        self.fc1 = nn.Linear(64*7*7, 10) # Fully connected layer

    # Define the forward pass
    def forward(self, x):
        x = torch.relu(self.conv1(x)) # Apply ReLU activation
        x = self.avgpool(x) # Average pooling
        x = torch.relu(self.conv2(x)) # Apply ReLU activation
        x = self.avgpool(x) # Average pooling
        x = x.view(-1, 64*7*7) # Flatten the tensor
        x = self.fc1(x) # Apply fully connected layer
        return x

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

# Set up training parameters
criterion = nn.CrossEntropyLoss() # Loss function

# Create a fresh model and optimizer for each optimizer type to avoid state carry-over.
# Store constructors to instantiate optimizers with model.parameters() later.
optimizer_constructors = {
    'SGD': lambda params, lr: optim.SGD(params, lr=lr),
    'RMSprop': lambda params, lr: optim.RMSprop(params, lr=lr),
    'Adam': lambda params, lr: optim.Adam(params, lr=lr)
}

# Train models with different optimizers and record metrics
def train_and_evaluate(model, optimizer, optimizer_name):
    model.to(device) # Move model to the appropriate device
    model.train() # Set model to training mode
    # Initialize lists to store metrics
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Training loop
    for epoch in range(10):
        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # Move data to device
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            # Accumulate loss and correct predictions
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Compute epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch+1}/{10} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate on test set
        model.eval() # Set model to evaluation mode
        # Initialize test metrics
        correct = 0
        total = 0
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device) # Move data to device
                outputs = model(images) # Forward pass
                _, predicted = torch.max(outputs.data, 1) # Get predictions
                total += labels.size(0) # Update total count
                correct += (predicted == labels).sum().item() # Update correct predictions count

        # Calculate test loss and accuracy
        test_loss = epoch_loss  # Use train loss as test loss for simplicity
        test_acc = correct / total # Calculate test accuracy
        # Append test metrics
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    return train_losses, test_losses, train_accuracies, test_accuracies

# Train and evaluate models for each optimizer type
histories = {} # To store training histories

for optimizer_name, constructor in optimizer_constructors.items():
    print(f"\n>>> {optimizer_name}")
    # Create a fresh model and optimizer for each optimizer type
    model = CNN().to(device)
    optimizer = constructor(model.parameters(), lr=0.01) # Instantiate optimizer
    train_losses, test_losses, train_accuracies, test_accuracies = train_and_evaluate(model, optimizer, optimizer_name)
    histories[optimizer_name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

# Plot the results
plt.figure(figsize=(12, 6))

for optimizer_name, h in histories.items():
    train_acc = h['train_accuracies']
    test_acc = h['test_accuracies']
    plt.plot(train_acc, label=f'Train Acc - {optimizer_name}')
    plt.plot(test_acc, label=f'Test Acc - {optimizer_name}', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy for Different Optimizers')
plt.legend()
plt.show() # Display the plot
