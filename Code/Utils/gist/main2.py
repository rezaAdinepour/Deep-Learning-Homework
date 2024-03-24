# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from SLP import Single_Layer_Perceptron

# check GPU availability
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



# Split the dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

# create new input feature x^2
X_val_new = torch.pow(X_val, 2)


print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("Validation set size:", len(X_val))


class0 = X_val[y_val.flatten() == 0]
class1 = X_val[y_val.flatten() == 1]

class0_new = X_val_new[y_val.flatten() == 0]
class1_new = X_val_new[y_val.flatten() == 1]


# plot dataset
plt.figure(figsize=(15, 7))

# plot training set
plt.subplot(1, 2, 1)
plt.scatter(X_val[:, 0].cpu().numpy(), X_val[:, 1].cpu().numpy(), c=y_val.cpu().numpy(), cmap='jet', marker='o')
plt.title("orginal data")
plt.xlabel('x')
plt.ylabel('y')

# plot test set
plt.subplot(1, 2, 2)
plt.scatter(X_val_new[:, 0].cpu().numpy(), X_val_new[:, 1].cpu().numpy(), c=y_val.cpu().numpy(), cmap='jet', marker='o')
plt.title("data in power of 2")
plt.xlabel('x')
plt.ylabel('y')

plt.show()



# set network parameters
num_of_inputs = 2
epochs = 50
learning_rate = 0.01
w = np.random.random(num_of_inputs + 1) - 0.5 # initial weights with random values between -0.5 and 0.5

perceptron = Single_Layer_Perceptron(num_of_inputs, epochs, learning_rate)

# Train phase
new_weight, F1_score, avg_loss, avg_acc = perceptron.train2(X_val.cpu().numpy(), y_val.cpu().numpy(), epochs)
print(f"Average loss: {avg_loss:.4f}, Average accuracy: {avg_acc:.4f}, F1 score: {F1_score:.4f}")