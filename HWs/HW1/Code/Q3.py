import torch
import numpy as np
import matplotlib.pyplot as plt
from SLP import single_layer_perceptron

# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# create x point
x = torch.linspace(-10, 10, 1000, device=device)
print(x.size())

# calculate f(x)
f = torch.sin(x) + 3 * torch.pow(x, 17) - 5 * torch.pow(x, 2)

# concatinare x and f
data = torch.concatenate((x.reshape(-1, 1), f.reshape(-1, 1)), dim=1)
print(data.size())

# plot f(x)
# plt.figure(figsize=(9, 7))
# plt.plot(x.cpu().numpy(), f.cpu().numpy(), linestyle='-', color='blue', linewidth=2)
# plt.xlabel('x', fontsize=14)
# plt.ylabel('f(x)', fontsize=14)
# plt.title('Plot of the function f(x)', fontsize=16)
# plt.grid(linestyle='--', linewidth=0.6)


f1 = torch.exp(x)
plt.figure(figsize=(9, 7))
plt.plot(x.cpu().numpy(), f1.cpu().numpy(), linestyle='-', color='blue', linewidth=2, label='$e^x$')
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.title('Plot of the $e^x$', fontsize=16)
plt.grid(linestyle='--', linewidth=0.6)



f_approx = 1 + x + (0.5 * torch.pow(x, 2)) + ((1/6) * torch.pow(x, 3)) + ((1/24) * torch.pow(x, 4))
plt.plot(x.cpu().numpy(), f_approx.cpu().numpy(), linestyle='-', color='red', linewidth=2, label="approximate")
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.title('Plot of the $e^x$', fontsize=16)
plt.grid(linestyle='--', linewidth=0.6)


plt.show()