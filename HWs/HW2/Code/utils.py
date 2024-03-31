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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(2, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        x = F.sigmoid(self.hidden1(x))
        x = F.sigmoid(self.hidden2(x))
        x = self.output(x)
        return F.softmax(x, dim=1)
    





# class MLP(nn.Module):
#     def __init__(self, input_neurons, hidden_layers, hidden_neurons, output_neurons):
#         super(MLP, self).__init__()
#         self.hidden_layers = nn.ModuleList()
        
#         # Input layer
#         self.hidden_layers.append(nn.Linear(input_neurons, hidden_neurons))
        
#         # Hidden layers
#         for _ in range(hidden_layers - 1):
#             self.hidden_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        
#         # Output layer
#         self.output = nn.Linear(hidden_neurons, output_neurons)

#     def forward(self, x):
#         for hidden_layer in self.hidden_layers:
#             x = F.relu(hidden_layer(x))
#         x = self.output(x)
#         return F.softmax(x, dim=1)