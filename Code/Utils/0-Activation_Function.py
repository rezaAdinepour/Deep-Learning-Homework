import torch
import matplotlib.pyplot as plt
import os
from GPU_Available import*


# clear terminal
clear = lambda: os.system('clear') # if you use windows, change 'clear' argument with 'cls'
clear()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check GPU
print_torch()
print_torch_gpu_info()


# create x points
x = torch.linspace(-10, 10, 100, device=device)


# create activation function
y = torch.zeros([8, 100], dtype=torch.float32, device=device)

y[0] = torch.nn.functional.sigmoid(x)
y[1] = torch.nn.functional.tanh(x)
y[2] = torch.nn.functional.relu(x)
y[3] = torch.nn.functional.leaky_relu(x, 0.5)
y[4] = torch.nn.functional.elu(x, 0.5)
y[5] = torch.nn.functional.selu(x)
y[6] = torch.nn.functional.gelu(x)
#y[7] = torch.autograd(y[6])
y[7] = torch.nn.functional.softplus(x, 1, 1)





# plot outputs
m = 2
n = 4
title = ['Sigmoid', 'Tanh', 'RelU', 'Leaky RelU', 'ELU', 'SELU', 'GELU', 'Soft plus']
color = ['black', 'blue', 'red', 'yellow', 'green', 'orange', 'purple', 'red']

plt.figure(figsize=(12, 9))
for i in range(len(title)):
    plt.subplot(m, n, i + 1)
    plt.plot(x.cpu().numpy(), y[i].cpu().numpy(), color=color[i])
    plt.title(title[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)



plt.show()