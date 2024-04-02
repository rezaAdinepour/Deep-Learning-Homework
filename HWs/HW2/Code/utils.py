import torch.nn as nn
import torch.nn.functional as F



# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.hidden = nn.Linear(2, 3)
#         self.output = nn.Linear(3, 2)

#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.output(x)
#         return F.softmax(x, dim=1)





# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size, bias=True, device="cuda:0")
#         #self.hidden2 = nn.Linear(10, 10)
#         self.layer2 = nn.Linear(hidden_size, output_size, bias=True, device="cuda:0")

#     def forward(self, x):
#         #x = F.sigmoid(self.hidden1(x))
#         #x = F.sigmoid(self.hidden2(x))
#         # x = self.output(x)
#         y_hidden = self.layer1(x)
#         y = self.layer2(F.relu(y_hidden))
#         return y
#         #return F.softmax(y, dim=1)
    



import torch

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, device='cpu'):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return F.softmax(x, dim=1)