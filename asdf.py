#pytorch imports
from matplotlib import transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import timm

#unused
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#medmnist imports
import medmnist
from medmnist import PathMNIST
from medmnist import INFO, Evaluator

#running on cpu
print(torch.__version__)
print(torch.cuda.is_available())


#line break
print("-----------")


#information about the dataset
info = INFO['pathmnist']
task = info['task']
print(task)
n_channels = info['n_channels']
print(n_channels)
n_classes = len(info['label'])
print(n_classes)
# 'label': {'0': 'adipose', 
#           '1': 'background', 
#           '2': 'debris', 
#           '3': 'lymphocytes', 
#           '4': 'mucus', 
#           '5': 'smooth muscle', 
#           '6': 'normal colon mucosa', 
#           '7': 'cancer-associated stroma', 
#           '8': 'colorectal adenocarcinoma epithelium'}

#transformations for the dataset
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

#importing the dataset
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)