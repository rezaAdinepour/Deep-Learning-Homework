import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC



# Define a kernel function to map the 2D data to 3D
def kernel_trick(x):
    return np.append(x, np.expand_dims(x[:, 0]**2 + x[:, 1]**2, axis=1), axis=1)


# Generate a linear non-separable 2-class dataset
X, y = make_circles(n_samples=100, factor=.3, noise=.05)

# Plot the original 2D data
plt.figure(figsize=(6,6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
plt.title('Original 2D data')
plt.show()


# Apply the kernel function to the data
X_3D = kernel_trick(X)

# Plot the 3D data
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c=y, s=50, cmap='viridis')
ax.set_title('Data after applying the kernel trick')

plt.show()