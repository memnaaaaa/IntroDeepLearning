# lab3/p4_regularisation.py
# This script implements and compares different regularization techniques
# (Dropout, Batch Normalization, and L2 Regularization) in a Convolutional Neural Network (CNN).

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

# Create a CNN model with Dropout, Batch Normalization, and L2 regularization
class RegularizedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, weight_decay=0.0): # dropout_rate and weight_decay for regularization
        super(RegularizedCNN, self).__init__() # Initialize the CNN model
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # First convolutional layer
        self.bn1 = nn.BatchNorm2d(32) # Batch normalization after first conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Second convolutional layer
        self.bn2 = nn.BatchNorm2d(64) # Batch normalization after second conv layer
        self.dropout = nn.Dropout(dropout_rate) # Dropout layer
        self.fc1 = nn.Linear(64*7*7, 10) # Fully connected layer
        self.weight_decay = weight_decay # Store weight decay for reference
    
    # Define the forward pass
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x))) # Conv1 + BN + ReLU
        x = torch.max_pool2d(x, 2) # Max pooling
        x = torch.relu(self.bn2(self.conv2(x))) # Conv2 + BN + ReLU
        x = torch.max_pool2d(x, 2) # Max pooling
        x = self.dropout(x) # Apply dropout
        x = x.view(-1, 64*7*7) # Flatten the tensor
        x = self.fc1(x) # Apply fully connected layer
        return x

    # Extra representation for debugging and logging purposes
    def extra_repr(self):
        return f"Dropout: {self.dropout.p}, Weight Decay: {self.weight_decay}"
    
# Set up training parameters
criterion = nn.CrossEntropyLoss() # Loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

# Train models with different regularization combinations and record metrics
def train_and_evaluate(model, optimizer, num_epochs=10):
    model.train() # Set model to training mode
    # Initialize lists to store metrics
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Training loop
    for epoch in range(num_epochs):
        # Initialize running loss and correct predictions count
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

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate on test set
        model.eval() # Set model to evaluation mode
        # Initialize test running loss and correct predictions count
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
        test_acc = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    return train_losses, test_losses, train_accuracies, test_accuracies

# Define different regularization combinations to test
regularization_combinations = [
    (0.5, 0.0),    # Dropout 0.5, no L2
    (0.5, 0.0001), # Dropout 0.5, L2 0.0001
    (0.0, 0.0),    # No dropout, no L2
    (0.0, 0.001),  # No dropout, L2 0.001
    (0.2, 0.0),    # Dropout 0.2, no L2
    (0.2, 0.0001)  # Dropout 0.2, L2 0.0001
]

# Train and evaluate models for each regularization combination
histories = {}

for dropout_rate, weight_decay in regularization_combinations:
    model = RegularizedCNN(dropout_rate=dropout_rate, weight_decay=weight_decay).to(device) # Initialize model
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=weight_decay) # Define optimizer with L2 regularization
    histories[(dropout_rate, weight_decay)] = train_and_evaluate(model, optimizer) # Train and evaluate the model

# Plot the results
plt.figure(figsize=(12, 6))

for (reg, hist) in histories.items():
    plt.plot(hist[0], label=f'Train Acc - {reg}')
    plt.plot(hist[2], label=f'Test Acc - {reg}')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy for Different Regularization Combinations')
plt.legend()
plt.show() # Display the plot
