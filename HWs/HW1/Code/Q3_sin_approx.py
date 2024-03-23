import torch
import matplotlib.pyplot as plt




x = torch.linspace(-10, 10, 1000)
y = torch.sin(x)
f_approx_sin = x - ((1/6) * torch.pow(x, 3)) + ((1/120) * torch.pow(x, 5)) - ((1/5040) * torch.pow(x, 7)) + ((1/362880) * torch.pow(x, 9))


plt.figure(figsize=(9, 7))
plt.plot(x, y, color="blue", linewidth=2, label="original $sin(x)$")
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-1, 1])
plt.xlim([-5, 5])
plt.grid(linestyle='--', linewidth=0.6)
plt.legend()


plt.figure(figsize=(9, 7))
plt.plot(x, f_approx_sin, color="red", linewidth=2, label="approx $sin(x)$")
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-1, 1])
plt.xlim([-5, 5])
plt.grid(linestyle='--', linewidth=0.6)
plt.legend()


plt.show()