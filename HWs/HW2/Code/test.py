import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import*
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.datasets import make_circles, make_classification, make_moons
from torch.optim import SGD
from torch.nn import BCELoss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x, y = datasets.make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)




# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert the numpy arrays to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)





# Initialize the MLP
model = MLP(input_size=2, hidden_size=10, output_size=2, device=device)

# Define the loss function and the optimizer
criterion = BCELoss()
optimizer = SGD(model.parameters(), lr=0.01)

y_train_onehot = F.one_hot(y_train)
y_test_onehot = F.one_hot(y_test)


# Lists to store loss and accuracy values
train_loss = []
train_acc = []
test_loss = []
test_acc = []

# Training phase
for epoch in range(100):  # number of epochs
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train_onehot.float())  # use one-hot encoded targets
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / y_train.size(0)

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
        print(f'Epoch {epoch}, Train Loss: {train_loss[-1]}, Train Accuracy: {train_acc[-1] * 100}%, Test Loss: {test_loss[-1]}, Test Accuracy: {test_acc[-1] * 100}%')


# Calculate predictions for training data
model.eval()
with torch.no_grad():
    outputs = model(x_train)
    _, train_predicted = torch.max(outputs, 1)



# Plot training and testing loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.title('Loss')

# Plot training and testing accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# Calculate confusion matrices
train_cm = confusion_matrix(y_train.cpu(), train_predicted.cpu())
test_cm = confusion_matrix(y_test.cpu(), predicted.cpu())

# Plot training confusion matrix
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.heatmap(train_cm, annot=True)
plt.title('Train Confusion Matrix')

# Plot testing confusion matrix
plt.subplot(1, 2, 2)
sns.heatmap(test_cm, annot=True)
plt.title('Test Confusion Matrix')

# Plot decision boundary
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

model.eval()
with torch.no_grad():
    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float).cuda())
    _, Z = torch.max(Z, 1)
    Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 4))
plt.contourf(xx, yy, Z.cpu(), alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', marker='o')
plt.show()


# # Save the trained model
# torch.save(model.state_dict(), 'model.pth')

# # Load the trained model
# model = MLP(input_size=2, hidden_size=10, output_size=2, device=device)
# model.load_state_dict(torch.load('model.pth'))

# # Predict the labels for the test set
# with torch.no_grad():
#     outputs = model(x_test)
#     _, predicted = torch.max(outputs, 1)
#     correct = (predicted == y_test).sum().item()
#     accuracy = correct / y_test.size(0)

#     print(f'Test Loss: {loss.item()}, Test Accuracy: {accuracy * 100}%')
    
# Evaluation phase
# model.eval()
# with torch.no_grad():
#     outputs = model(x_test)
#     loss = criterion(outputs, y_test_onehot.float())  # use one-hot encoded targets

#     # Calculate test accuracy
#     _, predicted = torch.max(outputs, 1)
#     correct = (predicted == y_test).sum().item()
#     accuracy = correct / y_test.size(0)

#     print(f'Test Loss: {loss.item()}, Test Accuracy: {accuracy * 100}%')