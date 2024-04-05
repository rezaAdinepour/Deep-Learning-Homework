import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import cv2



# multi layer perceptron class
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
    


class multi_layer_perceptron(nn.Module):
    def __init__(self):
        super(multi_layer_perceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.layers(x)
    


# function for load images from personal dir
def read_img(dir, format):
    images = []
    for img in os.listdir(dir):
        if img.endswith("." + format):
            images.append(cv2.imread(os.path.join(dir, img)))
    return len(images), images