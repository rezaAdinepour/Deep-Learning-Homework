import torch
import numpy as np
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



x = torch.linspace(-10, 10, 1000, device=device)
print(x.size())

f = torch.sin(x) + 3 * torch.pow(x, 17) - 5 * torch.pow(x, 2)


plt.figure(figsize=(9, 7))
plt.plot(x, f)


plt.show()