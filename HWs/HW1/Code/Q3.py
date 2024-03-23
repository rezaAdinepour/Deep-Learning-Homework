import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

# # Assume x and y are your data
# x = torch.randn(100, 1)  # 100 samples, 1 feature
# y = 2*x + 3  # simple linear relationship with noise

# # Define the model
# model = nn.Linear(1, 1)

# # Define the loss function
# criterion = nn.MSELoss()

# # Define the optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # Train the model
# for epoch in range(1000):
#     # Forward pass
#     output = model(x)
#     loss = criterion(output, y)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 100 == 0:
#         print(f'Epoch {epoch+1}/{1000}, Loss: {loss.item()}')

# # Get the final model predictions
# final_outputs = model(x).detach().numpy()

# # Plot the data and the fitted line
# plt.scatter(x, y, label='Data')
# plt.plot(x, final_outputs, label='Fitted line', color='red')
# plt.legend()
# plt.show()








import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the function to approximate
def target_function(x):
    return np.sin(x) + 3 * x**17 - 5 * x**2

# Generate training data
np.random.seed(0)
num_samples = 1000
x_train = np.linspace(-1, 1, num_samples)
y_train = target_function(x_train)

# Convert training data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set the dimensions of the model
input_size = 1
hidden_size = 100
output_size = 1

# Create an instance of the MLP model
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Set the number of epochs
num_epochs = 100

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate test data
x_test = np.linspace(-1, 1, num_samples)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)

# Compute predictions using the trained model
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)

# Convert predictions to numpy array
y_pred = y_pred_tensor.squeeze(1).numpy()

# Plot the results
plt.plot(x_test, target_function(x_test), label='Target Function')
plt.plot(x_test, y_pred, label='MLP Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()