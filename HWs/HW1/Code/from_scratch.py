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

# Split the dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)


plt.scatter(X_train[:, 0].cpu().numpy(), X_train[:, 1].cpu().numpy(), c=y_train.cpu().numpy(), cmap='jet', marker='.')
# plt.show()

num_of_inputs = 2
epochs = 100
learning_rate = 0.01
w = np.random.random(num_of_inputs + 1) - 0.5



perceptron = single_layer_perceptron(2, 1, 0.01)
perceptron.train(X_tensor.cpu().numpy(), y_tensor.cpu().numpy(), 1)

