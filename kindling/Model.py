import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Model(nn.Module):
    def __init__(self, device):
        super.__init__(Model, self)

        self.device = device
        self.optimizer = None
        self.criterion = None

        self.model = None

    def compile(self, optimizer, criterion):
        pass

    def fit(self, x, y):
        pass

