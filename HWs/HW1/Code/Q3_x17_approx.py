import torch
import matplotlib.pyplot as plt




x = torch.linspace(-10, 10, 1000)
y = 3 * torch.pow(x, 17) - 5 * torch.pow(x, 2)


plt.figure(figsize=(9, 7))
plt.plot(x, y, color="blue", linewidth=2, label="original $x^{17}$")
plt.xlabel('x')
plt.ylabel('y')

plt.grid(linestyle='--', linewidth=0.6)
plt.legend()



plt.show()