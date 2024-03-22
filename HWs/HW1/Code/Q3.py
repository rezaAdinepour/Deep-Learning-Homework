import torch
import numpy as np
import matplotlib.pyplot as plt


# check GPU a
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



x = torch.linspace(-10, 10, 1000, device=device)
print(x.size())

f = torch.sin(x) + 3 * torch.pow(x, 17) - 5 * torch.pow(x, 2)


plt.figure(figsize=(9, 7))

plt.plot(x.cpu().numpy(), f.cpu().numpy(), linestyle='-', color='blue', linewidth=2)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.title('Plot of the function f(x)', fontsize=16)
plt.grid(linestyle='--', linewidth=0.6)


plt.show()