import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split


class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()

        self.in_channels = in_channels

        kernal_size = 3  # Kernel size for MNIST
        if in_channels > 1:
            kernal_size = 5  # Kernel size for CIFAR10

        self.conv1 = nn.Conv2d(in_channels, 32, kernal_size)
        self.conv2 = nn.Conv2d(32, 64, kernal_size)
        self.fc1 = nn.Linear(64*5*5, 384)
        self.fc2 = nn.Linear(384, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Structured in LeNet
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#Function to get train and validation datasets.
def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset


def train(
        model,
        train_dataset,
        valid_dataset,
        device
):
    # Hyperparameters for training
    epochs = 5  # 3 epochs for MNIST is better
    batch_size = 50
    learning_rate = 0.001
    interval = 100

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs+1):
        model.train()
        print(f'Epoch {epoch}:')
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target) # Calculate loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Display the loss after each interval
            if i % interval == 0:
                print(f'\t[{i * len(data)}/{len(train_dataset)}]\tLoss: {loss.item():.4f}')

        # Following code for validation is mostly adapted from CIFAR10_Multiple_Linear_Regression.ipynb
        model.eval()
        correct = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Display the accuracy after each interval
        accuracy = 100. * correct / len(valid_loader.dataset)
        print(f'Accuracy: {accuracy:.2f}\n')

    results = dict(
        model=model
    )

    return results


def CNN(dataset_name, device):

    #CIFAR-10 has 3 channels whereas MNIST has 1.
    if dataset_name == "CIFAR10":
        in_channels= 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')

    model = Net(in_channels).to(device)

    train_dataset, valid_dataset = load_dataset(dataset_name)

    results = train(model, train_dataset, valid_dataset, device)

    return results
