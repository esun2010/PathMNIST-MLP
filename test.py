#pytorch (related) imports
import torch
import torch.nn as nn
import torchvision.models as models

#importing from model.py
from model import NeuralNetwork
from model import get_data_loader_test
from model import misc_info
from model import test_loop

#fetching model
model = NeuralNetwork()
checkpoint = torch.load('checkpoints/model_weights.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#testing the model
loss_fn = nn.CrossEntropyLoss()
test_loader = get_data_loader_test(batch_size=128, info=misc_info())
test_loop(test_loader, model, loss_fn)
print("Done!")