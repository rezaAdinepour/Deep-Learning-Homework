import torch
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



a = torch.tensor([1, 2, 3], dtype=torch.float64, device=device)
b = torch.tensor([3, 4, 5], dtype=torch.float64, device=device)
c1 = torch.add(a, b)
c2 = a + b
c3 = a.add_(b)

