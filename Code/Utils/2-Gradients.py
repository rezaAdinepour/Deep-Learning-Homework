import torch
import numpy as np
import matplotlib.pyplot as plt



device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


x = torch.linspace(-10, 10, 100, requires_grad=True, device=device)

y = torch.nn.functional.sigmoid(x)
y.backward(torch.ones_like(x))


plt.figure(figsize=(12, 9))
plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy(), color='red', label='Sigmoid')
plt.plot(x.cpu().detach().numpy(), x.grad.cpu().detach().numpy(), color='blue', label='Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()


plt.show()