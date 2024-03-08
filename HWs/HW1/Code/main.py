# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import imageio
import glob
from SLP import*


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# read dataset and remove first 8 row and set new header with this name: x, y, label
df = pd.read_csv("data.txt", skiprows=8, header=None, names=['x', 'y', "label"])
print("shape of data frame is:", df.shape)
print(df)


X = df[['x', 'y']]
y = df["label"]


# Convert the data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device).view(-1, 1)


# Split the dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42) 
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)


print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("Validation set size:", len(X_val))



# Divide the dataset into two classes
class0 = X_tensor[y_tensor.flatten() == 0]
class1 = X_tensor[y_tensor.flatten() == 1]



# Plot data
plt.figure(figsize=(9, 7))

plt.scatter(class0[:, 0].cpu().numpy(), class0[:, 1].cpu().numpy(), label="Class 0", marker='.')
plt.scatter(class1[:, 0].cpu().numpy(), class1[:, 1].cpu().numpy(), label="Class 1", marker='x')

plt.xlabel('x')
plt.ylabel('y')
plt.title("Dataset samples")
plt.legend()

plt.show()




model = Perceptron()
if torch.cuda.is_available():
    model = model.to('cuda')
    print("Model is using GPU")
else:
    print("Model is using CPU")


# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Create DataLoaders
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)


# Train the perceptron
for epoch in range(100):
    model.train()
    train_losses = []
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    print("Epoch: {}/100, Loss: {:.4f}".format(epoch+1, np.mean(train_losses)))
    

# Generate a grid of points
x_min, x_max = X_tensor[:, 0].cpu().min() - 1, X_tensor[:, 0].cpu().max() + 1
y_min, y_max = X_tensor[:, 1].cpu().min() - 1, X_tensor[:, 1].cpu().max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                    np.arange(y_min, y_max, 0.01))

# Predict the class for each point
grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
preds = model(grid)
Z = preds.view(xx.shape).detach().cpu().numpy()

# Plot the points and the decision boundary
plt.figure(figsize=(9, 7))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(class0[:, 0].cpu().numpy(), class0[:, 1].cpu().numpy(), label="Class 0", marker='.')
plt.scatter(class1[:, 0].cpu().numpy(), class1[:, 1].cpu().numpy(), label="Class 1", marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Dataset samples and decision boundary")
plt.legend()
plt.show()

    # # Plot the loss and save it as an image file
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training loss')
    # plt.title("Epoch: {}/100".format(epoch+1))
    # plt.legend()
    # plt.savefig('images/epoch_{}.png'.format(epoch+1))
    # plt.close()

# # Create a GIF from the image files
# images = []
# for filename in glob.glob('images/*.png'):
#     images.append(imageio.imread(filename))
# imageio.mimsave('training.gif', images)