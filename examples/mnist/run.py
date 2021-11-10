import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mnist_model import *
from mnist_data import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    model = MNISTModel(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.compile(optimizer, criterion)

    data = MNIST()
    data_gen = data.inf_generator(data.train_loader)

    model.fit(data_gen,
              data.train_eval_loader,
              data.test_loader,
              nepochs=20,
              batches_per_epoch=1000)

if __name__ == '__main__':
    main()

