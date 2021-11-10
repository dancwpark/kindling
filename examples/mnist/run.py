import numpy as np
import torch

from mnist_model import *
from mnist_data import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    model = MNISTModel(device)

    optimizer = optim.SGD(model.paramters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.compile(criterion, optimizer)

    data = MNIST()
    data_gen = data.inf_generator(data.train_loader)

    model.fit(data_gen,
              data.train_eval_loader,
              data.test_loader,
              20,
              1000)

if __name__ == '__main__':
    main()

