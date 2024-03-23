import torch
import numpy as np
import matplotlib.pyplot as plt
from SLP import single_layer_perceptron
import math



# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# create x point
x = torch.linspace(-10, 10, 1000, device=device)
print(x.size())

# calculate f(x)
f = torch.sin(x) + 3 * torch.pow(x, 17) - 5 * torch.pow(x, 2)
f_sin = torch.sin(x)
f_approx_sin = x - ((1/6) * torch.pow(x, 3)) + ((1/120) * torch.pow(x, 5)) - ((1/5040) * torch.pow(x, 7)) + ((1/362880) * torch.pow(x, 9))

# concatinare x and f
data = torch.concatenate((x.reshape(-1, 1), f.reshape(-1, 1)), dim=1)
print(data.size())


# plot f(x)

plt.figure(figsize=(9, 7))
plt.plot(x.cpu().numpy(), f_sin.cpu().numpy(), linestyle='-', color='blue', linewidth=2, label="orginal")
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
# plt.title('Plot of the function f(x)', fontsize=16)
plt.grid(linestyle='--', linewidth=0.6)

plt.plot(x.cpu().numpy(), f_approx.cpu().numpy(), linestyle='-', color='red', linewidth=2, label="approximate")
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
# plt.title('Plot of the function f(x)', fontsize=16)
plt.grid(linestyle='--', linewidth=0.6)
plt.legend()

# plt.figure(figsize=(9, 7))
# plt.plot(x.cpu().numpy(), f.cpu().numpy(), linestyle='-', color='blue', linewidth=2, label="orginal")
# plt.xlabel('x', fontsize=14)
# plt.ylabel('f(x)', fontsize=14)
# # plt.title('Plot of the function f(x)', fontsize=16)
# plt.grid(linestyle='--', linewidth=0.6)

# plt.plot(x.cpu().numpy(), f_approx.cpu().numpy(), linestyle='-', color='red', linewidth=2, label="approximate")
# plt.xlabel('x', fontsize=14)
# plt.ylabel('f(x)', fontsize=14)
# # plt.title('Plot of the function f(x)', fontsize=16)
# plt.grid(linestyle='--', linewidth=0.6)
# plt.legend()




# taylor approxiamtion of e^x
# f_approx = 0
# N = 20
# for n in range(N):
#     f_approx = f_approx + (torch.pow(x, n)) / (torch.math.factorial(n))

#     plt.cla()
#     plt.plot(x.cpu().numpy(), f_approx.cpu().numpy(), linestyle='-', color='red', linewidth=2, label="approximate")
#     plt.xlabel('x', fontsize=14)
#     plt.ylabel('f(x)', fontsize=14)
#     plt.title('Plot of the $e^x$', fontsize=16)
#     plt.grid(linestyle='--', linewidth=0.6)
#     print('n = {:2d}'.format(n))
#     plt.pause(1)

plt.show()



