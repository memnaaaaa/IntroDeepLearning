import numpy as np

# Create vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Compute dot product
dot_product = np.dot(v1, v2)

# Create matrix and vector
A = np.array([[1, 2],
              [3, 4]])
b = np.array([5, 6])

# Matrix - vector multiplication
result = np.matmul(A, b)

# Compute transpose of A
A_T = A.T

print("Dot product :", dot_product)
print("Matrix - vector multiplication :", result)
print("Transpose :\n", A_T)
