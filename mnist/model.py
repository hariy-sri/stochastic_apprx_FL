import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, SGD
from torch.optim.optimizer import required
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Type
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays


class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x1,x2,x3

class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.fc3(x2)  # No softmax or log-softmax here
        return x1, x2, x3


def train(net: nn.Module,
          trainloader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          epochs: int, 
          proximal_mu: float, 
          server_round: int, 
          stochastic: str, 
          method: str, 
          device: str,
          client_id: int,
          gradient_norms: list = None,
          batch_gradients: list = None):  # Now this is per-batch!
    criterion = torch.nn.CrossEntropyLoss()

    if stochastic == "per_epoch":
        if method == "single_dominant":
            if client_id == 0:
                lambda1 = lambda e: (1 / (((server_round - 1) * epochs + e + 10) ** (3/4)))
            else:
                lambda1 = lambda e: (1 / (((server_round - 1) * epochs + e + 1)))
        elif method == "double_dominant":
            if client_id == 0:
                lambda1 = lambda e: (1 / (((server_round - 1) * epochs + e + 10) ** (3/4)))
            if client_id == 1:
                lambda1 = lambda e: (1 / 2)*(1 / (((server_round - 1) * epochs + e + 10) ** (3/4)))
            else:
                lambda1 = lambda e: (1 / (((server_round - 1) * epochs + e + 1)))
        elif method == "no_dominant":
            lambda1 = lambda e: (1 / (((server_round - 1) * epochs + e + 1)))
        else:
            print("Either mention method as 'single_dominant' or 'double_dominant' or 'no_dominant' ")
            import sys
            sys.exit()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    net.to(device)
    global_params = [val.clone() for val in net.parameters()]
    net.train()

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            proximal_term = 0.0

            if proximal_mu > 0:
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)
                _, _, out = net(images)
                loss = criterion(out, labels) + (proximal_mu / 2) * proximal_term
            else:
                _, _, out = net(images)
                loss = criterion(out, labels)

            loss.backward()

            # # Track per-batch gradient norm (client 0 only)
            # if client_id == 0 and gradient_norms is not None:
            grad_norm = 0.0
            for param in net.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            gradient_norms.append(grad_norm)

           # Save flattened gradient vector
            if batch_gradients is not None:
                grads = [p.grad.view(-1).detach().cpu() for p in net.parameters() if p.grad is not None]
                batch_gradients.append(torch.cat(grads))

            optimizer.step()

        if epoch != epochs - 1 and stochastic == "per_epoch":
            scheduler.step()


def test(net: nn.Module, testloader: DataLoader, device: torch.device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            _,_,outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy