import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.randn(100) * 2 # noisy line

# Initialize parameters
w, b = 0.0, 0.0
eta = 0.01

# Gradient descent loop
losses = []
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
    if epoch % 20 == 0:
        print(f" Epoch {epoch}: w={w:.3f} , b={b:.3f} , loss ={loss:.3f}")

print(" Final parameters :", w , b )

# Plot loss curve
plt.plot(losses)
plt.xlabel(" Epoch ")
plt.ylabel("MSE Loss ")
plt.title(" Loss over Epochs ")
plt.show()

# Plot data and fitted line
plt.scatter(X, y, label=" Data ")
plt.plot(X, w * X + b, color="red", label=" Fitted Line ")
plt.legend()
plt.title(" Linear Regression Fit ")
plt.show()
