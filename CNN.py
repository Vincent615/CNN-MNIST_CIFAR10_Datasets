import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()

        if in_channels == 1:
            kernel_size = 3  # kernel size for MNIST
        else:
            kernel_size = 5  # kernel size for CIFAR10

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(1600, 384)
        self.fc2 = nn.Linear(384, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Structured in LeNet
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.fc3(x)
