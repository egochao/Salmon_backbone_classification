import torch
from torch import nn
import torchvision
from torchvision import models


class SalBinary(nn.Module):
    def __init__(self, model):
        super(SalBinary, self).__init__()
        self.classes = ["False", "True"]
        self.model = model
        self.dense = nn.Linear(1000, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        x = self.model(inp)
        x = self.dense(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    # bb = models.mobilenet_v3_large()
    bb = models.mobilenet_v3_small()
    # bb = models.mobilenet_v2()
    salmodel = SalBinary(bb)
    x = torch.rand((32, 3, 512, 512), requires_grad=True)

    x = salmodel(x)
    print(x.shape)