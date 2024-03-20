import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import*
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

X_tensor = X_tensor ** 2

# Split the dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)


plt.scatter(X_train[:, 0].cpu().numpy(), X_train[:, 1].cpu().numpy(), c=y_train.cpu().numpy(), cmap='jet', marker='.')
# plt.show()

num_of_inputs = 2
epochs = 500
learning_rate = 0.01
w = np.random.random(num_of_inputs + 1) - 0.5



perceptron = Single_Layer_Perceptron(num_of_inputs, epochs, learning_rate)
new_weight, F1_score, avg_loss, avg_acc = perceptron.train(X_tensor.cpu().numpy(), y_tensor.cpu().numpy(), epochs)
print(f"Average loss: {avg_loss:.4f}, Average accuracy: {avg_acc:.4f}, F1 score: {F1_score:.4f}")

print('-'*50)

# Test with validation set
val_loss, val_accuracy, val_f1 = perceptron.test(X_val.cpu().numpy(), y_val.cpu().numpy())
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

# Test with test set
test_loss, test_accuracy, test_f1 = perceptron.test(X_test.cpu().numpy(), y_test.cpu().numpy())
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")



