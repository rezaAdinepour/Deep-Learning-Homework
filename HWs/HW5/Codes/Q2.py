import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv("../inputs/dataset_1.csv")

# Shuffling dataset
df = shuffle(df)

# Split dataset into the inputs x and the outputs y (labels)
X = df[['x', 'y']]
y = df['label']

print("shape of data frame is:", df.shape)

X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device).view(-1, 1)

print("Training set size:", len(X_tensor))

class0 = torch.cat((X_tensor[y_tensor.flatten() == 0], y_tensor[y_tensor.flatten() == 0]), dim=1)
class1 = torch.cat((X_tensor[y_tensor.flatten() == 1], y_tensor[y_tensor.flatten() == 1]), dim=1)

plt.figure(figsize=(5, 5))

plt.scatter(class0[:, 0].cpu().numpy(), class0[:, 1].cpu().numpy(), label="Class 0", marker='.')
plt.scatter(class1[:, 0].cpu().numpy(), class1[:, 1].cpu().numpy(), label="Class 1", marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Dataset samples")
plt.legend()
plt.show()

# Determine number of samples per subset
num_subsets = 50
num_samples_per_class = min(len(class0), len(class1)) // num_subsets

subsets = []
for _ in range(num_subsets):
    indices_class0 = torch.randperm(len(class0))[:num_samples_per_class]
    indices_class1 = torch.randperm(len(class1))[:num_samples_per_class]
    
    subset_class0 = class0[indices_class0]
    subset_class1 = class1[indices_class1]
    
    subset = torch.cat((subset_class0, subset_class1), dim=0)
    subsets.append(subset)

for i, subset in enumerate(subsets):
    print(f"subset {i+1}: {subset.shape}")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GRUOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUOptimizer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Correct the input size for GRUOptimizer
input_size = 2  # Size of each of gradient and parameter (1 + 1)
hidden_size = 128
output_size = 1
gru_optimizer = GRUOptimizer(input_size, hidden_size, output_size).to(device)

optimizer_gru = optim.Adam(gru_optimizer.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Define evaluation function
def evaluate_model(f, data_loader):
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = f(X_batch)
            loss = criterion(outputs, y_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    error = loss.item()
    return accuracy, error

accuracies = []
errors = []

# Loop through each subset and optimize using `g`
for subset in subsets:
    # Split subset into training and testing sets with 2:8 ratio
    subset = subset.cpu().numpy()
    train_size = int(0.2 * len(subset))
    train_data, test_data = train_test_split(subset, train_size=train_size, random_state=42)
    
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    
    train_X, train_y = train_data[:, :2], train_data[:, 2].view(-1, 1)
    test_X, test_y = test_data[:, :2], test_data[:, 2].view(-1, 1)
    
    # Create DataLoader for the test set
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Outer loop
    for _ in range(10):
        f = MLP().to(device)
        optimizer_f = optim.SGD(f.parameters(), lr=0.01)
        
        hidden = torch.zeros(1, 1, hidden_size).to(device)  # Initialize hidden state
        # Inner loop
        for epoch in range(5):
            optimizer_f.zero_grad()
            
            # Forward pass
            outputs = f(train_X)
            loss = criterion(outputs, train_y)
            
            # Backward pass
            loss.backward()
            
            # Get the gradients
            gradients = []
            params = []
            for param in f.parameters():
                gradients.append(param.grad.view(-1, 1))
                params.append(param.view(-1, 1))
            gradients = torch.cat(gradients).view(1, -1, 1)
            params = torch.cat(params).view(1, -1, 1)
            
            # Concatenate gradients and parameters along the correct dimension
            grad_param_cat = torch.cat((gradients, params), dim=2).to(device)
            
            # Update the parameters using the GRU optimizer
            updates, hidden = gru_optimizer(grad_param_cat, hidden)
            updates = updates.view(-1)
            
            # Apply the updates to the parameters
            with torch.no_grad():
                for param, update in zip(f.parameters(), updates):
                    param -= 0.01 * update  # Learning rate could be adjusted
            
            # Zero the gradients after updating
            for param in f.parameters():
                param.grad = None
        
        # Evaluate network on testing set data
        f.eval()
        accuracy, error = evaluate_model(f, test_loader)
        accuracies.append(accuracy)
        errors.append(error)

# Report the average accuracy and error
avg_accuracy = sum(accuracies) / len(accuracies)
avg_error = sum(errors) / len(errors)

print(f"Average Accuracy: {avg_accuracy:.2f}%")
print(f"Average Error: {avg_error:.4f}")

# Repeat for dataset_2
df_2 = pd.read_csv("../inputs/dataset_2.csv")

# Shuffling dataset
df_2 = shuffle(df_2)

# Split dataset into the inputs x and the outputs y (labels)
X_2 = df_2[['x', 'y']]
y_2 = df_2['label']

print("shape of data frame is:", df_2.shape)

X2_tensor = torch.tensor(X_2.values, dtype=torch.float32).to(device)
y2_tensor = torch.tensor(y_2.values, dtype=torch.float32).to(device).view(-1, 1)

print("Training set size:", len(X2_tensor))

class0_2 = torch.cat((X2_tensor[y2_tensor.flatten() == 0], y2_tensor[y2_tensor.flatten() == 0]), dim=1)
class1_2 = torch.cat((X2_tensor[y2_tensor.flatten() == 1], y2_tensor[y2_tensor.flatten() == 1]), dim=1)

plt.figure(figsize=(5, 5))

plt.scatter(class0_2[:, 0].cpu().numpy(), class0_2[:, 1].cpu().numpy(), label="Class 0", marker='.')
plt.scatter(class1_2[:, 0].cpu().numpy(), class1_2[:, 1].cpu().numpy(), label="Class 1", marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Dataset samples")
plt.legend()
plt.show()

# Determine number of samples per subset
num_subsets_2 = 30
num_samples_per_class_2 = min(len(class0_2), len(class1_2)) // num_subsets_2

subsets_2 = []
for _ in range(num_subsets_2):
    indices_class0_2 = torch.randperm(len(class0_2))[:num_samples_per_class_2]
    indices_class1_2 = torch.randperm(len(class1_2))[:num_samples_per_class_2]
    
    subset_class0_2 = class0_2[indices_class0_2]
    subset_class1_2 = class1_2[indices_class1_2]
    
    subset_2 = torch.cat((subset_class0_2, subset_class1_2), dim=0)
    subsets_2.append(subset_2)

for i, subset in enumerate(subsets_2):
    print(f"subset {i+1}: {subset.shape}")

accuracies_2 = []
errors_2 = []

# Loop through each subset and optimize using `gru_optimizer`
for subset in subsets_2:
    # Split subset into training and testing sets with 2:8 ratio
    subset = subset.cpu().numpy()
    train_size = int(0.2 * len(subset))
    train_data, test_data = train_test_split(subset, train_size=train_size, random_state=42)
    
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    
    train_X, train_y = train_data[:, :2], train_data[:, 2].view(-1, 1)
    test_X, test_y = test_data[:, :2], test_data[:, 2].view(-1, 1)
    
    # Create DataLoader for the test set
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Outer loop
    for _ in range(10):
        f = MLP().to(device)
        optimizer_f = optim.SGD(f.parameters(), lr=0.01)
        
        hidden = torch.zeros(1, 1, hidden_size).to(device)  # Initialize hidden state
        # Inner loop
        for epoch in range(5):
            optimizer_f.zero_grad()
            
            # Forward pass
            outputs = f(train_X)
            loss = criterion(outputs, train_y)
            
            # Backward pass
            loss.backward()
            
            # Get the gradients
            gradients = []
            params = []
            for param in f.parameters():
                gradients.append(param.grad.view(-1, 1))
                params.append(param.view(-1, 1))
            gradients = torch.cat(gradients).view(1, -1, 1)
            params = torch.cat(params).view(1, -1, 1)
            
            # Concatenate gradients and parameters along the correct dimension
            grad_param_cat = torch.cat((gradients, params), dim=2).to(device)
            
            # Update the parameters using the GRU optimizer
            updates, hidden = gru_optimizer(grad_param_cat, hidden)
            updates = updates.view(-1)
            
            # Apply the updates to the parameters
            with torch.no_grad():
                for param, update in zip(f.parameters(), updates):
                    param -= 0.01 * update  # Learning rate could be adjusted
            
            # Zero the gradients after updating
            for param in f.parameters():
                param.grad = None
        
        # Evaluate network on testing set data
        f.eval()
        accuracy, error = evaluate_model(f, test_loader)
        accuracies_2.append(accuracy)
        errors_2.append(error)

# Report the average accuracy and error for dataset_2
avg_accuracy_2 = sum(accuracies_2) / len(accuracies_2)
avg_error_2 = sum(errors_2) / len(errors_2)

print(f"Average Accuracy for dataset_2: {avg_accuracy_2:.2f}%")
print(f"Average Error for dataset_2: {avg_error_2:.4f}")

