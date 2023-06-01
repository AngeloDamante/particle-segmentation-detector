"""Implementation of UNET Model"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from utils.definitions import DEPTH


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # first conv 3x3 ReLu
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv 3x3 ReLu
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=DEPTH, out_channels=1, features=None):
        super(UNET, self).__init__()
        if features is None: features = [64, 128, 256, 512]

        # operations
        self.ups = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part \
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck _
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up part /
        for feature in reversed(features):
            self.up_conv.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # we're going down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # and now we are in the bottom part
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # we're going up
        for idx in range(0, len(self.ups)):
            x = self.up_conv[idx](x)
            skip_connection = skip_connections[idx]

            # concatenate
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=False)
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # double conv
            x = self.ups[idx](concat_skip)

        # finally, the last convolution
        return self.final_conv(x)


def test():
    x = torch.randn((3, DEPTH, 512, 512))
    model = UNET(in_channels=DEPTH, out_channels=DEPTH)
    # print(model)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
