import torch
import torch.nn as nn
from config import IMG_SIZE, NUM_CLASSES

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, residual_conv, first_stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride= first_stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.residual_conv = residual_conv
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.residual_conv is not None:
            out += self.residual_conv(x)
        else:
            out += x
        return self.act2(out)

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(in_channels=64, out_channels = 64, num_repeats=2)
        self.layer2 = self._make_layer(in_channels=64, out_channels = 128, num_repeats=2)
        self.layer3 = self._make_layer(in_channels=128, out_channels = 256, num_repeats=2)
        self.layer4 = self._make_layer(in_channels=256, out_channels = 512, num_repeats=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x
    

    def _make_layer(self, in_channels, out_channels, num_repeats):
        layers = []
        if in_channels == out_channels:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # [8,64,56,56]
            for _ in range(num_repeats):
                layers.append(BasicBlock(out_channels, out_channels, padding=1, residual_conv=None))
        else:
            residual_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=1), 
                nn.BatchNorm2d(out_channels))
            layers.append(BasicBlock(in_channels, out_channels, padding=1, residual_conv=residual_conv, first_stride=2))
            for _ in range(num_repeats-1):
                layers.append(BasicBlock(out_channels, out_channels, padding=1, residual_conv=None))

        return nn.Sequential(*layers)

if __name__ == '__main__':
    BATCH_SIZE = 8
    NUM_CLASSES = 3
    input_tensor = torch.rand((BATCH_SIZE, 3, *IMG_SIZE))
    net = ResNet(num_classes=NUM_CLASSES)
    output_tensor = net(input_tensor)
    print(output_tensor.shape)