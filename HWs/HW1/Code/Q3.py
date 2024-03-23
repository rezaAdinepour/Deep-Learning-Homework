import torch
import numpy as np
import matplotlib.pyplot as plt
from SLP import single_layer_perceptron
import math




def taylor_approximation(x, n):
    # Taylor series for sin(x) around 0
    sin_x = sum(((-1)**i * x**(2*i+1)) / torch.math.factorial(2*i+1) for i in range(n))

    # Taylor series for x^17 around 0
    x_17 = x**17  # Since we're expanding around 0, this is just x^17

    # Taylor series for x^2 around 0
    x_2 = x**2  # Since we're expanding around 0, this is just x^2

    # Combine the series
    f_taylor = sin_x + 3*x_17 - 5*x_2

    return f_taylor




# check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# create x point
x = torch.linspace(-10, 10, 1000, device=device)
print(x.size())

# calculate f(x)
f = torch.sin(x) + 3 * torch.pow(x, 17) - 5 * torch.pow(x, 2)
f_approx = taylor_approximation(x, 1)
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



