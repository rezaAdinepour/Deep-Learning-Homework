import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import*
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.datasets import make_circles, make_classification, make_moons








device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x, y = datasets.make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Convert the numpy arrays to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)



# plt.figure(figsize=(9, 7))
# plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), c=y.cpu().numpy(), edgecolors='k', marker='o', s=50)
# plt.show()



model = MLP(input_size=2, hidden_size=2, output_size=2, device=device)

# Define the loss function and the optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

y_train_onehot = F.one_hot(y_train)
y_test_onehot = F.one_hot(y_test)
print(model)

# Specify the number of epochs
epochs = 500




# # Initialize lists to save the loss and accuracy values
# loss_values = []
# accuracy_values = []


# Lists to store loss and accuracy values
train_loss = []
train_acc = []
test_loss = []
test_acc = []


for epoch in range(epochs):
    model.train()
    # Forward pass
    outputs = model(x_train)
    
    # Compute loss
    loss = criterion(outputs, y_train_onehot.float())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / y_train.size(0)
    
    # Compute accuracy
    _, predicted = torch.max(outputs.data, 1)
    total = y_train.size(0)
    correct = (predicted == y_train.long()).sum().item()
    accuracy = correct / total

    # Store loss and accuracy
    train_loss.append(loss.item())
    train_acc.append(accuracy)


    # Evaluation phase
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        loss = criterion(outputs, y_test_onehot.float())  # use one-hot encoded targets

        # Calculate test accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / y_test.size(0)

        # Store loss and accuracy
        test_loss.append(loss.item())
        test_acc.append(accuracy)
        

    # Print loss and accuracy every 10 epochs
    if epoch % 10 == 0:
        print ('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}, Test Loss: {:.4f}, Test Accuracy: {:.2f} '.format(epoch+1, epochs, train_loss[-1], train_acc[-1], test_loss[-1], test_acc[-1]))


# Calculate predictions for training data
_, train_predicted = torch.max(model(x_train).data, 1)

# Set a consistent figure size
plt.figure(figsize=(10, 10))

# Plot loss values
plt.subplot(2, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy values
plt.subplot(2, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()



# Calculate confusion matrices
train_cm = confusion_matrix(y_train.cpu().numpy(), train_predicted.cpu().numpy())
test_cm = confusion_matrix(y_test.cpu().numpy(), predicted.cpu().numpy())

# Plot training confusion matrix
plt.subplot(2, 2, 3)
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Train Confusion Matrix')
plt.xlabel('Predicted class')
plt.ylabel('True class')

# Plot testing confusion matrix
plt.subplot(2, 2, 4)
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted class')
plt.ylabel('True class')


# Generate a grid of points
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min.item(), x_max.item(), 0.1),
                     np.arange(y_min.item(), y_max.item(), 0.1))

model.eval()
with torch.no_grad():
    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float).to(device))
    _, Z = torch.max(Z, 1)
    Z = Z.reshape(xx.shape)

# Plot the decision regions
plt.figure(figsize=(9, 7))
plt.contourf(xx, yy, Z.cpu().numpy(), alpha=0.8, cmap='RdBu')
plt.scatter(x_train[:, 0].cpu().numpy(), x_train[:, 1].cpu().numpy(), c=y_train.cpu().numpy(), cmap='RdBu', edgecolors='k', marker='o', s=50)
plt.title('Decision regions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Adjust the layout for better readability
plt.tight_layout()
plt.show()