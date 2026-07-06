#pytorch (related) imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data

#medmnist imports
import medmnist
from medmnist import PathMNIST
from medmnist import INFO, Evaluator





def misc_info():
    #running on cpu
    print("Version: " + torch.__version__)
    print("CUDA available: " + str(torch.cuda.is_available()))

    info = INFO['pathmnist']
    task = info['task']
    print("Task: " + task)
    n_channels = info['n_channels']
    print("Number of channels: " + str(n_channels))
    n_classes = len(info['label'])
    print("Number of classes: " + str(n_classes))
    # 'label': {'0': 'adipose', 
    #           '1': 'background', 
    #           '2': 'debris', 
    #           '3': 'lymphocytes', 
    #           '4': 'mucus', 
    #           '5': 'smooth muscle', 
    #           '6': 'normal colon mucosa', 
    #           '7': 'cancer-associated stroma', 
    #           '8': 'colorectal adenocarcinoma epithelium'}

    #returning info dictionary for later use
    return info


def get_data_loader_train(batch_size, info):
    #transformations for the dataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    #importing the dataset
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    #data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*batch_size, shuffle=False)
    return train_loader, train_loader_at_eval

def get_data_loader_test(batch_size, info):
    #transformations for the dataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    #importing the dataset
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', transform=data_transform, download=True)
    #data loaders
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


class NeuralNetwork(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.flatten = nn.Flatten()
        seq_modules = nn.Sequential(
            nn.Linear(3*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        self.seq_modules = seq_modules

    def forward(self, x):
        x = self.flatten(x)
        logits = self.seq_modules(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer,batch_size=128):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        y = y.squeeze(1).long() 
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            y = y.squeeze(1).long() 
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

