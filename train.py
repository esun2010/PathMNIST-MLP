#pytorch (related) imports
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

#importing from model.py
from model import NeuralNetwork
from model import get_data_loader_train
from model import misc_info
from model import train_loop
from model import test_loop

import os

#hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 10

#gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#getting the data loaders and misc info
info = misc_info()
train_loader, train_loader_at_eval = get_data_loader_train(batch_size=batch_size, info=info)


#instantiating the model, loss function, and optimizer
loss_fn = nn.CrossEntropyLoss()
model = NeuralNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Resume from checkpoint if one exists
checkpoint_path = 'checkpoints/model_weights.pth'
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

#training loop
for t in range(start_epoch, start_epoch + epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer, device, batch_size=batch_size)
    test_loop(train_loader_at_eval, model, loss_fn, device)

torch.save({
    'epoch': t,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_path)    
print("Done!")