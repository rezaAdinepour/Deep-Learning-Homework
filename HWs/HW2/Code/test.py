import torch
import matplotlib.pyplot as plt
from utils import*
from sklearn import datasets




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x, y = datasets.make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)
x = torch.tensor(x, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.float32, device=device)


plt.figure(figsize=(9, 7))
plt.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), c=y.cpu().numpy(), cmap="rainbow")
# plt.show()




model = MLP().to(device)
criterion = nn.BCEWithLogitsLoss()


print(model)