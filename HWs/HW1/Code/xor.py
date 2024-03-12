import numpy as np
import torch
import matplotlib.pyplot as plt
from SLP import*


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = torch.tensor([ [0, 0], [0, 1], [1, 0], [1, 1] ], device=device)
label = torch.tensor([0, 1, 1, 0])
print(data.shape)
print(data)

fig1 = plt.figure(figsize=(9, 7))
plt.scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), c=label.cpu().numpy())
# plt.show()

num_of_inputs = 2
epochs = 50
lr = 0.01
w = np.random.random(num_of_inputs + 1) - 0.5


perceptron = single_layer_perceptron(num_of_inputs, epochs, lr)
_, loss_history, accuracy_history = perceptron.train(data.cpu().numpy() ** 2, label.cpu().numpy(), epochs)


plt.show()