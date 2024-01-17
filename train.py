import torch
from torchvision import transforms, datasets

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from CNN import CNN


torch.multiprocessing.set_sharing_strategy('file_system')


class Params():
    def __init__(self):
        #self.dataset = "MNIST"
        self.dataset = "CIFAR10"
        self.device = 'cuda'
        #self.device = 'cpu'

        self.batch_size = 50
        self.epochs = 5  # 3 epochs for MNIST, 5 epochs for CIFAR10
        self.lr = 0.001
        self.interval = 100  # display interval

params = Params()


def load_dataset(dataset_name=params.dataset, batch_size=params.batch_size):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.MNIST('./data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader


def train(model, train_loader, valid_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=params.lr)  # Use Adam optimizer

    for epoch in range(1, params.epochs+1):
        # Training
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
            if i % params.interval == 0:
                print(f'\t[{i * len(data)}/{len(train_loader)*params.batch_size}]\tLoss: {loss.item():.4f}')

        # Validation
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

    return model


def test(model, test_loader, device):
    model.eval()
    num_correct = 0
    total = 0
    for i, (data, targets) in enumerate(test_loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

    acc = float(num_correct) / total
    return acc


def main():
    device = torch.device("cuda" if params.device != "cpu" and torch.cuda.is_available() else "cpu")

    # CIFAR-10 has 3 channels whereas MNIST has 1
    if params.dataset == "CIFAR10":
        in_channels= 3
    elif params.dataset == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {params.dataset}')

    train_loader, valid_loader, test_loader = load_dataset()

    net = CNN(in_channels).to(device)
    model = train(net, train_loader, valid_loader, device)

    accuracy = test(model, test_loader, device)

    print(f"Result on {params.dataset}:")
    print(f"\taccuracy : {accuracy}")


if __name__ == "__main__":
    main()
