import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import*
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x, y = datasets.make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)
x = torch.tensor(x, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.float32, device=device)


# plt.figure(figsize=(9, 7))
# plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), c=y.cpu().numpy(), edgecolors='k', marker='o', s=50)
#plt.show()




model = MLP().to(device)


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the loss function
criterion = nn.CrossEntropyLoss()
print(model)


# Specify the number of epochs
epochs = 100

# Convert labels to one-hot encoding
y_onehot = F.one_hot(y.long(), num_classes=2)



# Initialize lists to save the loss and accuracy values
loss_values = []
accuracy_values = []

for epoch in range(epochs):
    # Forward pass
    outputs = model(x)
    
    # Compute loss
    loss = criterion(outputs, y.long())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save loss value
    loss_values.append(loss.item())
    
    # Compute accuracy
    _, predicted = torch.max(outputs.data, 1)
    total = y.size(0)
    correct = (predicted == y.long()).sum().item()
    accuracy = correct / total

    # Save accuracy value
    accuracy_values.append(accuracy)
    
    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}'.format(epoch+1, epochs, loss.item(), accuracy))





# Set a consistent figure size
plt.figure(figsize=(15, 9))

# Plot loss values
plt.subplot(2, 2, 1)
plt.plot(loss_values, label='Loss', color='red')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot accuracy values
plt.subplot(2, 2, 2)
plt.plot(accuracy_values, label='Accuracy', color='green')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Compute confusion matrix
_, predicted = torch.max(model(x).data, 1)
cm = confusion_matrix(y.cpu().numpy(), predicted.cpu().numpy())

# Plot confusion matrix
plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted class')
plt.ylabel('True class')

# Generate a grid of points
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min.item(), x_max.item(), 0.1),
                     np.arange(y_min.item(), y_max.item(), 0.1))

# Predict the class for each point
grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
preds = model(grid)
_, predicted = torch.max(preds.data, 1)

# Reshape the predicted classes to have the same shape as xx
Z = predicted.cpu().numpy().reshape(xx.shape)

# Plot the decision regions
plt.subplot(2, 2, 4)
plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdBu')
plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), c=y.cpu().numpy(), cmap='RdBu', edgecolors='k', marker='o', s=50)
plt.title('Decision regions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Adjust the layout for better readability
plt.tight_layout()
plt.show()