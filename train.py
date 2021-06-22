import torch
import torch.optim as optim
from torchvision import models
from torch import nn
from icecream import ic

from model import SalBinary
from dataloader import SalmonDataset, LabelProc, trans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def val(dataset_val, net):
    true_acc = 0
    total = 0
    for data in enumerate(dataset_val):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = net(inputs)



def train(epoch):
    # transform = trans()
    transform = None
    labelproc = LabelProc(ty="binary", threshold=0, func=max)
    dataset = SalmonDataset("pre_load_3c.pk", labelproc, transform=transform)

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [1000, 198],
                                                               generator=torch.Generator().manual_seed(42))
    batch_size = 32
    ic(len(dataset))
    ic(len(dataset_train))
    ic(len(dataset_val))
    dataset_train = torch.utils.data.DataLoader(dataset_train,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=4)

    dataset_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=4)

    # bb = models.mobilenet_v3_large()
    # bb = models.mobilenet_v3_small()
    # bb = models.mobilenet_v2()
    bb = models.mobilenet_v3_small()
    net = SalBinary(bb)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for ep in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(dataset_train):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                ic('[%d, %5d] loss: %.3f' %
                   (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == '__main__':
    train(10)
