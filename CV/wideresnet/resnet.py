import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, shorcut=False, dropout=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_size)

        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_size)

        if shorcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_size),
            )
        else:
            self.shortcut = nn.Sequential()  # Empty sequential equals to empty layer.

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))  # SHORCUT
        return out


class WideResNet(nn.Module):
    def __init__(self, i_channels=3, o_channels=64, scale_factor=1, n_classes=10):
        super(WideResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            i_channels,
            o_channels * scale_factor,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(o_channels * scale_factor)

        self.blocks = nn.Sequential(
            ResBlock(
                o_channels * scale_factor,
                o_channels * scale_factor,
                1,
            ),
            ResBlock(
                o_channels * scale_factor,
                o_channels * scale_factor,
                1,
            ),
            ResBlock(
                o_channels * scale_factor,
                o_channels * 2 * scale_factor,
                2,
                shorcut=True,
            ),
            ResBlock(
                o_channels * 2 * scale_factor,
                o_channels * 2 * scale_factor,
                1,
            ),
            ResBlock(
                o_channels * 2 * scale_factor,
                o_channels * 4 * scale_factor,
                2,
                shorcut=True,
            ),
            ResBlock(
                o_channels * 4 * scale_factor,
                o_channels * 4 * scale_factor,
                1,
            ),
            ResBlock(
                o_channels * 4 * scale_factor,
                o_channels * 8 * scale_factor,
                2,
                shorcut=True,
            ),
            ResBlock(
                o_channels * 8 * scale_factor,
                o_channels * 8 * scale_factor,
                1,
            ),
        )
        self.fw = nn.Linear(o_channels * 8 * scale_factor, n_classes)  # 10 Classes

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fw(out)
        return out
