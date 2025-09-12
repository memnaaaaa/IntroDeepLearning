import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.randn(100) * 2 # noisy line

# Test different learning rates
etas = [0.001, 0.01, 0.1, 0.5, 0.864]
loss_curves = {}

# Gradient descent for each learning rate
for eta in etas:
    # Initialize parameters
    w, b = 0.0, 0.0
    losses = []

    # Gradient descent loop
    for epoch in range(200):
        y_pred = w * X + b
        error = y - y_pred

        # Compute gradients
        dw = -2 * np.mean(X * error)
        db = -2 * np.mean(error)

        # Update parameters
        w = w - eta * dw
        b = b - eta * db

        # Compute loss
        loss = np.mean(error ** 2)
        losses.append(loss)
    loss_curves[eta] = losses

# Plot loss curves for each learning rate
plt.figure(figsize=(8, 5))
for eta in etas:
    plt.plot(loss_curves[eta], label=f"η = {eta}")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss over Epochs for Different Learning Rates")
plt.legend()
plt.show()
