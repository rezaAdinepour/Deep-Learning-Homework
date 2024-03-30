import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return F.softmax(x, dim=1)