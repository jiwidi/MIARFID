"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, shorcut=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(
            out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_size)

        if shorcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=False,),
                nn.BatchNorm2d(out_size),
            )
        else:
            self.shortcut = nn.Sequential()  # Empty sequential equals to empty layer.

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))  # SHORCUT
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_size = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.blocks = nn.Sequential(
            ResBlock(64, 64, 1, shorcut=False),
            ResBlock(64, 64, 1, shorcut=False),
            ResBlock(64, 128, 2, shorcut=True),
            ResBlock(128, 128, 1, shorcut=False),
            ResBlock(128, 256, 2, shorcut=True),
            ResBlock(256, 256, 1, shorcut=False),
            ResBlock(256, 512, 2, shorcut=True),
            ResBlock(512, 512, 1, shorcut=False),
        )
        self.fw = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fw(out)
        return out


def ResNet18():
    return ResNet()


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
