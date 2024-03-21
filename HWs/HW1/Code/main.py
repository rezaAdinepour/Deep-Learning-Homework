# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from SLP import single_layer_perceptron
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.preprocessing import PolynomialFeatures





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# read dataset and remove first 8 row and set new header with this name: x, y, label
df = pd.read_csv("data.txt", skiprows=8, header=None, names=['x', 'y', "label"])
print("shape of data frame is:", df.shape)
print(df)


X = df[['x', 'y']]
y = df["label"]


# # Convert the data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device).view(-1, 1)


# # Create a PolynomialFeatures object
# poly = PolynomialFeatures(degree=3, include_bias=True)

# # Transform the features
# X_poly = poly.fit_transform(X)

# # Convert the data to PyTorch tensors
# X_tensor = torch.tensor(X_poly, dtype=torch.float32).to(device)






# Split the dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)


print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("Validation set size:", len(X_val))



# Divide the dataset into two classes
class0 = X_tensor[y_tensor.flatten() == 0]
class1 = X_tensor[y_tensor.flatten() == 1]

class0_test = X_test[y_test.flatten() == 0]
class1_test = X_test[y_test.flatten() == 1]

class0_val = X_val[y_val.flatten() == 0]
class1_val = X_val[y_val.flatten() == 1]



# Plot data
plt.figure(figsize=(25, 7))

plt.subplot(1, 3, 1)
plt.scatter(class0[:, 0].cpu().numpy(), class0[:, 1].cpu().numpy(), label="Class 0", marker='.')
plt.scatter(class1[:, 0].cpu().numpy(), class1[:, 1].cpu().numpy(), label="Class 1", marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Dataset samples")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(class0_test[:, 0].cpu().numpy(), class0_test[:, 1].cpu().numpy(), label="Class 0", marker='.')
plt.scatter(class1_test[:, 0].cpu().numpy(), class1_test[:, 1].cpu().numpy(), label="Class 1", marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Testing set samples")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(class0_val[:, 0].cpu().numpy(), class0_val[:, 1].cpu().numpy(), label="Class 0", marker='.')
plt.scatter(class1_val[:, 0].cpu().numpy(), class1_val[:, 1].cpu().numpy(), label="Class 1", marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Validation set samples")
plt.legend()

plt.show()





    


# Initialize the model
model = single_layer_perceptron(X_train.size(1)).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Convert labels to integer type
y_train = y_train.long()
y_test = y_test.long()
y_val = y_val.long()

# Initialize lists to store losses and accuracies
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Train the model
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train.float())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    predicted = torch.round(outputs.data)
    correct = (predicted == y_train).sum().item()
    train_accuracy = correct / y_train.size(0)
    train_f1 = f1_score(y_train.cpu().numpy(), predicted.cpu().numpy())

    # Calculate loss and accuracy for validation set
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val.float())
    val_predicted = torch.round(val_outputs.data)
    val_correct = (val_predicted == y_val).sum().item()
    val_accuracy = val_correct / y_val.size(0)
    val_f1 = f1_score(y_val.cpu().numpy(), val_predicted.cpu().numpy())

    # Store losses and accuracies
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss.item())
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}")

print('-'*80)
print(f"Validation Loss: {sum(val_losses) / len(val_losses):.4f}, Validation Accuracy: {sum(val_accuracies) / len(val_accuracies):.4f}, Validation F1 Score: {val_f1:.4f}")


# Plot training loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend()

print("-"*50)


# Test phase
test_outputs = model(X_test)
test_loss = criterion(test_outputs, y_test.float())
test_predicted = torch.round(test_outputs.data)
test_correct = (test_predicted == y_test).sum().item()
test_accuracy = test_correct / y_test.size(0)
test_f1 = f1_score(y_test.cpu().numpy(), test_predicted.cpu().numpy())
test_cm = confusion_matrix(y_test.cpu().numpy(), test_predicted.cpu().numpy())




# Calculate F1 score and confusion matrix for training set
train_predicted = torch.round(model(X_train).data)
train_f1 = f1_score(y_train.cpu().numpy(), train_predicted.cpu().numpy())
train_cm = confusion_matrix(y_train.cpu().numpy(), train_predicted.cpu().numpy())

# Calculate F1 score and confusion matrix for validation set
val_cm = confusion_matrix(y_val.cpu().numpy(), val_predicted.cpu().numpy())



print(f"Test loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")
print(f"Validation loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}")
print("Final weights:", model.fc.weight.data)



# Plot confusion matrices
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.heatmap(train_cm, annot=True, fmt='d', ax=axs[0])
axs[0].set_title('Train Confusion Matrix')

sns.heatmap(val_cm, annot=True, fmt='d', ax=axs[1])
axs[1].set_title('Validation Confusion Matrix')

sns.heatmap(test_cm, annot=True, fmt='d', ax=axs[2])
axs[2].set_title('Test Confusion Matrix')



plt.show()













# model = Perceptron()
# model = model.to('cuda')



# # Define the loss function and the optimizer
# criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# EPOCH = 50


# # Create DataLoaders
# train_data = TensorDataset(X_train, y_train)
# val_data = TensorDataset(X_val, y_val)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

# # Initialize lists to store losses and accuracies


# # Train the perceptron
# for epoch in range(EPOCH):
#     model.train()
#     train_losses = []
#     train_accuracies = []
#     for i, (inputs, labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(inputs)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())

#         # Calculate accuracy
#         model.eval()
#         with torch.no_grad():
#             train_output = model(X_train)
#         train_preds = (train_output > 0.5).float()
#         train_acc = accuracy_score(y_train.cpu().numpy(), train_preds.cpu().numpy())

#         # Store losses and accuracies
#         train_losses.append(np.mean(train_losses))
#         train_accuracies.append(train_acc)




#         # Get the weights of the model
#         w = list(model.parameters())
#         w0 = w[0].data.cpu().numpy()
#         w1 = w[1].data.cpu().numpy()

#     # Compute the line equation
#     line_x = np.linspace(-10, 10, 100)
#     line_y = (-w1 - w0[0][0]*line_x) / w0[0][1]

#     # Plot the points and the decision boundary
#     plt.cla()
#     plt.scatter(X_train[:, 0].cpu().numpy(), X_train[:, 1].cpu().numpy(), c=y_train[:, 0].cpu().numpy(), cmap='jet', marker='.')
#     plt.plot(line_x, line_y)
#     plt.xlim(-10, 10)
#     plt.ylim(-10, 10)
#     plt.text(-10, 11, 'epoch = {:2d}'.format(epoch), fontdict={'size': 14, 'color':  'black'})
#     plt.pause(0.0001)

#     print(f"Epoch: {epoch+1}/{EPOCH}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}")

#     # print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, EPOCH, np.mean(train_losses)))
#     # print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, EPOCH, np.mean(train_losses)))
    

# # Generate a grid of points
# x_min, x_max = X_tensor[:, 0].cpu().min() - 1, X_tensor[:, 0].cpu().max() + 1
# y_min, y_max = X_tensor[:, 1].cpu().min() - 1, X_tensor[:, 1].cpu().max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                     np.arange(y_min, y_max, 0.01))

# # Predict the class for each point
# grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
# preds = model(grid)
# Z = preds.view(xx.shape).detach().cpu().numpy()

# # Plot the points and the decision boundary
# plt.figure(figsize=(9, 7))
# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(class0[:, 0].cpu().numpy(), class0[:, 1].cpu().numpy(), label="Class 0", marker='.')
# plt.scatter(class1[:, 0].cpu().numpy(), class1[:, 1].cpu().numpy(), label="Class 1", marker='x')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title("Dataset samples and decision boundary")
# plt.legend()


# # Plot losses
# plt.figure(figsize=(9, 7))
# plt.plot(train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # Plot accuracies
# plt.figure(figsize=(9, 7))
# plt.plot(train_accuracies, label='Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()



# # Compute confusion matrix for training set
# model.eval()
# with torch.no_grad():
#     train_output = model(X_train)
# train_preds = (train_output > 0.5).float()
# cm = confusion_matrix(y_train.cpu().numpy(), train_preds.cpu().numpy())

# # Plot confusion matrix
# plt.figure(figsize=(9, 7))
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('True')






# plt.show()
