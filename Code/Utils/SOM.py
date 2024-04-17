import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


X, y = datasets.make_blobs(n_samples=100, centers=[(0, 0), (0, 1), (0.5, 1.5), (1, 2), (1.5, 1.5), (2, 1)], cluster_std=0.1)
    
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet_r')
plt.title("Input data")
plt.xlabel("X1")
plt.ylabel("X2")




# initial weigth
w = np.random.rand(2, 6) - 0.5



plt.show()