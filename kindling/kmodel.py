import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from kindling.utils import *

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()

        self.device = device
        self.optimizer = None
        self.criterion = None

        self.model = None

    def compile(self, optimizer, criterion):
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

    def fit(self, data_gen,
            train_eval_loader,
            test_loader,
            nepochs=20,
            batches_per_epoch=1000):
        for itr in range(nepochs * batches_per_epoch):
            self.optimizer.zero_grad()
            x, y = data_gen.__next__()
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.forward(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            if itr % batches_per_epoch == 0:
                with torch.no_grad():
                    train_acc = self.accuracy(train_eval_loader)
                    val_acc = self.accuracy(test_loader)
                    print("Epoch {:04d} \
                           | Train Acc {:.4f} \
                           | Test Acc {:.4f}".format(itr//batches_per_epoch, 
                                                     train_acc, 
                                                     val_acc))

    def accuracy(self, dataset_loader):
        total_correct = 0
        for x, y in dataset_loader:
            x = x.to(self.device)
            # Should not be 10, make general
            y = one_hot(np.array(y.numpy()), 10)
            target_class = np.argmax(y, axis=1)
            predicted_class = np.argmax(
                    self.forward(x).cpu().detach().numpy(),
                    axis=1)
            total_correct += np.sum(predicted_class == target_class)
        return total_correct / len(dataset_loader.dataset)

    def to_device(self):
        self.to(self.device)

