import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

# Assume x and y are your data
x = torch.tensor([0])  # 100 samples, 1 feature
y = torch.tensor([4]) # simple linear relationship with noise



# Define the model
model = nn.Linear(1, 1)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{1000}, Loss: {loss.item()}')

# Get the final model predictions
final_outputs = model(x).detach().numpy()

# Plot the data and the fitted line
plt.scatter(x, y, label='Data')
plt.plot(x, final_outputs, label='Fitted line', color='red')
plt.legend()
plt.show()