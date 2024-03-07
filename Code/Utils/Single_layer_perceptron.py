import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import torch
import numpy as np
import time

# Code execution on CPU
start_time_cpu = time.time()

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def predict(inputs, weights):
    inputs = inputs.to(weights.dtype)  # Convert inputs to the same data type as weights
    summation = torch.dot(inputs, weights[1:]) + weights[0]
    activation = 1.0 if (summation > 0.0) else 0.0
    return activation

x, y = datasets.make_blobs(n_samples=50, centers=[(-1, -1), (1, 1)], cluster_std=0.5)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='jet_r')

num_of_inputs = 2
epochs = 100
learning_rate = 0.1
w = torch.randn(num_of_inputs + 1, device=device) - 0.5

x = torch.tensor(x, device=device)
y = torch.tensor(y, device=device)

for epoch in range(epochs):
    fail_count = 0
    i = 0

    for inputs, label in zip(x, y):
        i = i + 1
        prediction = predict(inputs, w.to(device))  # Move weights tensor to the same device as inputs

        if (label != prediction):
            line_x = np.linspace(-3, 3, 100)  # Define line_x as an array of values from -3 to 3
            inputs_cpu = inputs.reshape(inputs.shape[0]).cpu()  # Move inputs tensor to CPU device
            inputs_gpu = inputs_cpu.to(device)  # Move inputs to the GPU
            w[1:] += learning_rate * (label - prediction) * inputs_gpu
            w[0] += learning_rate * (label - prediction)
            fail_count += 1

            plt.cla()
            plt.scatter(x.cpu()[:, 0], x.cpu()[:, 1], c=y.cpu(), cmap='jet_r')
            line_x = torch.tensor(line_x, device=w.device)
            line_y = (-w[0] - w[1] * line_x) / w[2]
            plt.plot(line_x.cpu().numpy(), line_y.cpu().numpy())
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.text(0, -2.7, 'epoch|iter = {:2d}|{:2d}'.format(epoch, i), fontdict={'size': 14, 'color':  'blue'})
            plt.pause(0.1)

            

    if (fail_count == 0):
        end_time_cpu = time.time()
        execution_time_cpu = end_time_cpu - start_time_cpu
        plt.show()
        break



# Create a grid of points
x_values = np.linspace(-3, 3, 200)
y_values = np.linspace(-3, 3, 200)
xx, yy = np.meshgrid(x_values, y_values)

# Calculate the class for each point on the grid
grid = np.c_[xx.ravel(), yy.ravel()]
grid = torch.tensor(grid, device=device)
w = w.double()  # Convert w to Double
predictions = (torch.sigmoid(torch.matmul(grid, w[1:]) + w[0]) > 0.5).float()# Reshape the predictions to have the same shape as the grid
predictions = predictions.view(xx.shape)

# Plot the filled contour plot
plt.contourf(xx, yy, predictions.cpu().numpy(), cmap='jet_r', alpha=0.3)

# Plot the data points
plt.scatter(x.cpu()[:, 0], x.cpu()[:, 1], c=y.cpu(), cmap='jet_r')

# Plot the decision boundary
line_x = torch.linspace(-3, 3, 200, device=device)
line_y = (-w[0] - w[1] * line_x) / w[2]
plt.plot(line_x.cpu().numpy(), line_y.cpu().numpy(), 'k')

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()