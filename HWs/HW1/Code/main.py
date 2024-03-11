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
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns



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
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)


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
model = model.to('cuda')



# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
EPOCH = 2


# Create DataLoaders
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

# Initialize lists to store losses and accuracies


# Train the perceptron
for epoch in range(EPOCH):
    model.train()
    train_losses = []
    train_accuracies = []
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Calculate accuracy
        model.eval()
        with torch.no_grad():
            train_output = model(X_train)
        train_preds = (train_output > 0.5).float()
        train_acc = accuracy_score(y_train.cpu().numpy(), train_preds.cpu().numpy())

        # Store losses and accuracies
        train_losses.append(np.mean(train_losses))
        train_accuracies.append(train_acc)




        # Get the weights of the model
        w = list(model.parameters())
        w0 = w[0].data.cpu().numpy()
        w1 = w[1].data.cpu().numpy()

        # Compute the line equation
        line_x = np.linspace(-10, 10, 100)
        line_y = (-w1 - w0[0][0]*line_x) / w0[0][1]

        # Plot the points and the decision boundary
        plt.cla()
        plt.scatter(X_train[:, 0].cpu().numpy(), X_train[:, 1].cpu().numpy(), c=y_train[:, 0].cpu().numpy(), cmap='jet', marker='.')
        plt.plot(line_x, line_y)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.text(-10, 11, 'epoch|iter = {:2d}|{:2d}'.format(epoch, i), fontdict={'size': 14, 'color':  'black'})
        plt.pause(0.0001)

    print(f"Epoch: {epoch+1}/{EPOCH}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}")

    # print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, EPOCH, np.mean(train_losses)))
    # print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, EPOCH, np.mean(train_losses)))
    

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


# Plot losses
plt.figure(figsize=(9, 7))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracies
plt.figure(figsize=(9, 7))
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()



# Compute confusion matrix for training set
model.eval()
with torch.no_grad():
    train_output = model(X_train)
train_preds = (train_output > 0.5).float()
cm = confusion_matrix(y_train.cpu().numpy(), train_preds.cpu().numpy())

# Plot confusion matrix
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')






plt.show()










