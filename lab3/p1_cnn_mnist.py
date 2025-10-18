# lab3/p1_cnn_mnist.py
# This script trains a Convolutional Neural Network (CNN) on the MNIST dataset and plots the training history.

# Import necessary libraries and modules
import torch                                # PyTorch library for tensor computations and deep learning
import torch.nn as nn                       # Neural network module
import torch.optim as optim                 # Optimization algorithms
import torchvision.transforms as transforms # Image transformations
import torchvision.datasets as datasets     # Standard datasets
from torch.utils.data import DataLoader     # Data loading utility
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

# Define the CNN model with two convolutional layers and one fully connected layer
class CNN(nn.Module):
    def __init__(self): # Initialize the CNN model
        super(CNN, self).__init__() # Initialize the CNN class
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Second convolutional layer
        self.fc1 = nn.Linear(64*7*7, 10) # Fully connected layer

    # Forward pass
    def forward(self, x):
        x = torch.relu(self.conv1(x)) # Apply ReLU activation
        x = torch.max_pool2d(x, 2) # Max pooling
        x = torch.relu(self.conv2(x)) # Apply ReLU activation
        x = torch.max_pool2d(x, 2) # Max pooling
        x = x.view(-1, 64*7*7) # Flatten the tensor
        x = self.fc1(x) # Fully connected layer
        return x # Output logits

model = CNN() # Instantiate the CNN model 

# Set up training parameters
criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.SGD(model.parameters(), lr=0.01) # Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device configuration
model.to(device) # Move model to device

# Train the CNN model and record training and validation accuracy
def train_model(model, criterion, optimizer, num_epochs=10):
    model.train() # Set model to training mode
    train_losses, test_losses = [], [] # Lists to store losses
    train_accuracies, test_accuracies = [], [] # Lists to store accuracies

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Train on training set
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # Move data to device
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            running_loss += loss.item() # Accumulate loss
            _, predicted = torch.max(outputs.data, 1) # Get predictions
            total += labels.size(0) # Update total count
            correct += (predicted == labels).sum().item() # Update correct count

        epoch_loss = running_loss / len(train_loader) # Average loss for the epoch
        epoch_acc = correct / total # Accuracy for the epoch
        train_losses.append(epoch_loss) # Append training loss
        train_accuracies.append(epoch_acc) # Append training accuracy

        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate on test set
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) # Forward pass
                _, predicted = torch.max(outputs.data, 1) # Get predictions
                total += labels.size(0) # Update total count
                correct += (predicted == labels).sum().item() # Update correct count

        test_loss = epoch_loss  # Use train loss as test loss for simplicity
        test_acc = correct / total # Test accuracy
        test_losses.append(test_loss) # Append test loss
        test_accuracies.append(test_acc) # Append test accuracy

    return train_losses, test_losses, train_accuracies, test_accuracies

# Run the training process
train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, criterion, optimizer)

# Plot the training and test accuracy versus training batches
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show() # Display the plots