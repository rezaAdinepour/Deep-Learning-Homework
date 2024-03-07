# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# read dataset and remove first 8 row and set new header with this name: x, y, label
df = pd.read_csv("data.txt", skiprows=8, header=None, names=['x', 'y', 'label'])
print("shape of data frame is:", df.shape)
print(df)


X = df[['x', 'y']]
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("Validation set size:", len(X_val))


# divide this dataset into two classes
class0 = df[df["label"] == 0]
class1 = df[df["label"] == 1]


# plot data
plt.figure(figsize=(9, 7))

plt.scatter(class0['x'], class0['y'], label='Class 0', marker='.')
plt.scatter(class1['x'], class1['y'], label='Class 1', marker='x')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset samples')
plt.legend()

plt.show()