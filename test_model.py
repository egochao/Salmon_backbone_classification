import torch
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck

from model import SalBinary
from dataloader import SalmonDataset, LabelProc, trans
from backbones import ResNetBackBone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# bb = models.mobilenet_v3_small()
bb = ResNetBackBone(BasicBlock, [2, 2, 2, 2], [2, 2, 2, 2], groups=1,
                    width_per_group=64, in_channel=1)

net = SalBinary(bb)
net.to(device)

inp = torch.randn((4, 1, 244, 244))
inp = inp.to(device)

x = net(inp)
print(x.shape)