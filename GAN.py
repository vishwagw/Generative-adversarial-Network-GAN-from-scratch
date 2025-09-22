# creating the GAN:
# importing the libraries:
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# deciding the device to run the model on:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
