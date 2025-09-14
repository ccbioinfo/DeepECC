import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

parameter_choice = {
    'RU_1': [(32, 11, 1, 1)],
    'RU_2': [(32, 11, 1, 1), (32, 11, 1, 1)],
    'RU_3': [(32, 11, 1, 1), (32, 11, 1, 1), (32, 11, 4, 1)],
    'RU_4': [(32, 11, 1, 1), (32, 11, 1, 1), (32, 11, 4, 1), (32, 11, 4, 1)],
}


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, groups=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               dilation=dilation, padding=(kernel_size - 1)*dilation//2, groups=groups)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               dilation=dilation, padding=(kernel_size - 1)*dilation//2, groups=groups)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class ResidualNet(nn.Module):
    def __init__(self, channels, kernels, dilations, gourps):
        super(ResidualNet, self).__init__()
        self.top_block = nn.Conv1d(channels, channels, kernel_size=1, dilation=1, padding=0)
        self.res_block = nn.ModuleList()
        for i in range(2):
            self.res_block.append(ResidualBlock(channels, kernels, dilations, groups=gourps))

    def forward(self, x, res):
        res_ = self.top_block(x)
        for layer in self.res_block:
            x = layer(x)
        res += res_

        return x, res



class BreakModel(nn.Module):
    def __init__(self, seq_len=80, num_classes=2, parameter='Small', dropout_rate=0.1):
        super(BreakModel, self).__init__()
        N = parameter_choice[parameter][0][0]
        self.init_conv = nn.Conv1d(in_channels=4, out_channels=N, kernel_size=1, dilation=1, padding=0)

        self.residual_block = nn.ModuleList()
        for i in parameter_choice[parameter]:
            self.residual_block.append(ResidualNet(i[0], i[1], i[2], i[3]))

        self.inner_conv = nn.Conv1d(in_channels=N, out_channels=N, kernel_size=1, dilation=1, padding=0)
        self.final_cov = nn.Conv1d(N, 4, kernel_size=1, dilation=1, padding=0)

        self.max_pooling = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.flattening = nn.Flatten()
        in_features = int(seq_len / 5)
        in_features_2 = int(in_features / 2)
        in_features_4 = int(in_features / 4)
        self.linear_1 = nn.Linear(in_features, in_features_2)
        self.linear_2 = nn.Linear(in_features_2, in_features_4)
        self.linear_3 = nn.Linear(in_features_4, num_classes)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.init_conv(x)
        for i, layers in enumerate(self.residual_block):
            if i == 0:
                x, res = layers(x, 0)
            else:
                x, res = layers(x, res)
        x = self.inner_conv(x)
        x = self.final_cov(x+res)
        x = self.flattening(self.max_pooling(x))
        x = self.drop(F.leaky_relu(self.linear_1(x)))
        x = self.drop(F.leaky_relu(self.linear_2(x)))
        x = self.linear_3(x).squeeze(-1)
        return x


if __name__ == '__main__':
    model = BreakModel(seq_len=1600, num_classes=1, parameter='Big')

    x = torch.randn(5, 4, 1600)
    target = torch.tensor([1, 0, 1, 0, 1])
    print(x.shape)
    print(target.shape)
    output = model(x)
    print(output.shape)
    output = F.sigmoid(output)
    print(output)







