import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)
y = ( X [:, 0] + X [:, 1] > 0).astype(int) # label : above / below diagonal

# Initialize parameters
w = np.zeros(2)
b = 0.0
eta = 0.1

def sigmoid(z):
# Implement sigmoid
    return 1 / (1 + np.exp(-z))

losses = []
# Gradient descent loop
for epoch in range(200):
# Compute predictions
    z = X @ w + b
    y_pred = sigmoid(z)

    # Compute gradients
    dw = X.T @ (y_pred - y) / y.size
    db = np.mean(y_pred - y)

    # Update parameters
    w = w - eta * dw
    b = b - eta * db

    # Compute loss
    loss = - np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    losses.append(loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss ={loss:.3f}")

print("Final parameters:", w, b)

# Plot loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Loss over Epochs")
plt.show()

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = sigmoid(grid @ w + b).reshape(xx.shape)

plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="RdBu")
plt.title("Logistic Regression Decision Boundary")
plt.show()
