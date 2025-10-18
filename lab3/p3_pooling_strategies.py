# lab3/p3_pooling_strategies.py
# This script compares different pooling strategies (Max Pooling and Average Pooling)
# in a Convolutional Neural Network (CNN) on the MNIST dataset.

#  Import necessary libraries and modules
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

# Create a function to generate CNN models with different pooling strategies:
class CNN(nn.Module):
    def __init__(self, pooling_cls): # pooling_cls is the pooling layer class (nn.MaxPool2d or nn.AvgPool2d)
        super(CNN, self).__init__() # Initialize the CNN model
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Second convolutional layer
        # Instantiate pooling layer with kernel size 2
        self.pool = pooling_cls(2) # Pooling layer
        self.fc1 = nn.Linear(64*7*7, 10) # Fully connected layer

    # Define the forward pass
    def forward(self, x):
        x = torch.relu(self.conv1(x)) # Apply first convolution and ReLU activation
        x = self.pool(x) # Apply pooling
        x = torch.relu(self.conv2(x)) # Apply second convolution and ReLU activation
        x = self.pool(x) # Apply pooling
        x = x.view(-1, 64*7*7) # Flatten the tensor
        x = self.fc1(x) # Apply fully connected layer
        return x

# Define the pooling strategies
POOLING_MAP = { # Mapping of pooling strategy names to their corresponding classes
    'max': nn.MaxPool2d,
    'avg': nn.AvgPool2d
}

# Function to get model based on pooling type
def get_model(pooling_type):
    return CNN(POOLING_MAP[pooling_type]).to(device)

# Set up training parameters
criterion = nn.CrossEntropyLoss() # Loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

# Train and evaluate models with different pooling strategies
def train_and_evaluate(model, optimizer, num_epochs=10):
    # Lists to store losses and accuracies
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train() # Set model to training mode
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
        epoch_acc = correct / total # Calculate accuracy
        # Store losses and accuracies
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'Epoch {epoch}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate on test set
        model.eval() # Set model to evaluation mode
        # Initialize test running loss and correct predictions count
        test_running_loss = 0.0
        correct = 0
        total = 0
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device) # Move data to device
                outputs = model(images) # Forward pass
                loss = criterion(outputs, labels) # Compute loss
                test_running_loss += loss.item() # Accumulate test loss
                _, predicted = torch.max(outputs.data, 1) # Get predictions
                total += labels.size(0) # Update total count
                correct += (predicted == labels).sum().item() # Update correct predictions count

        # Calculate test loss and accuracy
        test_loss = test_running_loss / len(test_loader)
        test_acc = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    return train_losses, test_losses, train_accuracies, test_accuracies

# Main execution: Train and evaluate models with different pooling strategies
histories = {} # To store training histories
pooling_strategies = ['max', 'avg'] # Pooling strategies to evaluate

# Train and evaluate models for each pooling strategy
for strategy in pooling_strategies:
    model = get_model(strategy) # Get model with specified pooling strategy
    optimizer = optim.SGD(model.parameters(), lr=0.01) # Define optimizer
    histories[strategy] = train_and_evaluate(model, optimizer) # Train and evaluate the model

# Plot the results
plt.figure(figsize=(12, 6))

for strategy in pooling_strategies:
    # histories[strategy] -> (train_losses, test_losses, train_accs, test_accs)
    train_accs = histories[strategy][2]
    test_accs = histories[strategy][3]
    # convert to percentages for plotting
    plt.plot([a * 100.0 for a in train_accs], label=f'Train Acc - {strategy}')
    plt.plot([a * 100.0 for a in test_accs], '--', label=f'Test Acc - {strategy}')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy for Different Pooling Strategies')
plt.legend()
plt.show() # Display the plot
