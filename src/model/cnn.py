import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 25)

    def forward(self, x):
        # bsx28x28x1
        x = self.conv1(x)
        # bsx24x24x10
        x = F.relu(F.max_pool2d(x, kernel_size=2))
        # bsx12x12x10
        x = self.conv2_drop(self.conv2(x))
        # bsx8x8x20
        x = F.relu(F.max_pool2d(x, kernel_size=2))
        # bsx4x4x20
        x = x.view(-1, 320)
        # bsx320
        x = F.relu(self.fc1(x))
        # bsx50
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # bsx10

        return F.log_softmax(x, dim=1)
