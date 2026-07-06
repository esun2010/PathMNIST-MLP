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


#hyperparameters
learning_rate = 0.005
batch_size = 64
epochs = 10

#getting the data loaders and misc info
info = misc_info()
train_loader, train_loader_at_eval = get_data_loader_train(batch_size=batch_size, info=info)



loss_fn = nn.CrossEntropyLoss()
model = NeuralNetwork()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer, batch_size=batch_size)
    test_loop(train_loader_at_eval, model, loss_fn)
print("Done!")

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'checkpoints/model_weights.pth')