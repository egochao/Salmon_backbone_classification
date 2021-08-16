import os
import torch
import torch.optim as optim
from torchvision import models
from torch import nn
import numpy as np
from icecream import ic
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models.resnet import BasicBlock, Bottleneck

from model import SalBinary
from dataloader import SalmonDataset, LabelProc, trans
from backbones import ResNetBackBone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def val(dataset_val, net):
    correct = 0
    total = 0
    list_label = list()
    list_pred = list()
    for i, data in enumerate(dataset_val):
        inputs, labels = data
        inputs = inputs.to(device)
        out = net(inputs)
        _, pred = torch.max(out.data, 1)
        pred = pred.cpu().numpy()
        # total += labels.size(0)
        labels = labels.numpy()
        list_label.extend(labels)
        list_pred.extend(pred)
        # correct += (pred == labels).sum().item()
        # print(pred)
        # print(labels)

    # count_tr = list()
    # for tr, pr in zip(list_label, list_pred):
    #     print(tr, pr)
    #

    cf = confusion_matrix(list_label, list_pred)
    report = classification_report(list_label, list_pred)
    print(cf)
    print(report)
    # print(correct, total)
    # print(correct/total)


def train(epoch, test=False, threshold=1):
    # transform = trans()
    transform = None
    # labelproc = LabelProc(ty="binary", threshold=0, func=max)
    # dataset = SalmonDataset("pre_load_3c.pk", labelproc, transform=transform)
    # labelproc = LabelProc(ty="lks", threshold=threshold, func=max)
    labelproc = LabelProc(ty="lks-mid", threshold=threshold, func=max)
    dataset = SalmonDataset("pre_load_150x600.pk", labelproc, transform=transform)

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [1000, 198],
                                                               generator=torch.Generator().manual_seed(42))
    batch_size = 32
    ic(len(dataset))
    ic(len(dataset_train))
    ic(len(dataset_val))
    dataset_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=2)

    dataset_val = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=2)

    # bb = models.mobilenet_v3_large()
    bb = ResNetBackBone(BasicBlock, [2, 2, 2, 2], [2, 2, 2, 2], groups=1,
                        width_per_group=64, in_channel=1)

    net = SalBinary(bb)
    net.to(device)

    if os.path.exists("last_model.pth"):
        net.load_state_dict(torch.load("last_model.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    net.eval()
    val(dataset_val, net)
    net.train()

    if not test:
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
                    print(loss.item())

            # with torch.no_grad():
            net.eval()
            val(dataset_val, net)
            net.train()
            torch.save(net.state_dict(), "last_model.pth")


if __name__ == '__main__':
    train(100, test=False)
