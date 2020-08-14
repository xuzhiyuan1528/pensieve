import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
import os
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, dataset_dir):
        self.np_dir = dataset_dir
        self.input_pathes = os.path.join(self.np_dir, 'sample-state.npy')
        self.label_pathes = os.path.join(self.np_dir, 'sample-action.npy')
        self.inputs = np.load(self.input_pathes)
        self.labels = np.load(self.label_pathes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6*8, 36)
        self.bn1 = nn.BatchNorm1d(36)
        self.fc2 = nn.Linear(36, 12)
        self.bn2 = nn.BatchNorm1d(12)
        self.fc3 = nn.Linear(12, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        output = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, loss_f, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        # loss = F.nll_loss(output, target)
        target = target.squeeze()
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, loss_f):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            target = target.squeeze()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_f(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 20
    lr = 0.001
    gamma = 0.7
    use_cuda = False
    seed = 2020
    log_interval = 10
    save_model = True
    loss_f = nn.CrossEntropyLoss()
    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    if use_cuda:
        train_kwargs.update({'num_workers': 1,
                       'pin_memory': True},
                     )

    test_kwargs = {'batch_size': test_batch_size, 'shuffle': False}
    if use_cuda:
        test_kwargs.update({'num_workers': 1,
                       'pin_memory': True},
                     )

    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)

    dataset_dir = './best-rl'
    train_dataset = Dataset(dataset_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(train_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_f, log_interval)
        test(model, device, test_loader, loss_f)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), os.path.join(dataset_dir, 'best-rl_sup_model.pt'))

if __name__ == '__main__':
    main()