import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
import timm


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
    def forward(self, x: Tensor) -> Tensor:
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            res = self.downsample(res)
        out += res
        out = F.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.expansion = 4
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(res_block, 3, 64)
        self.layer2 = self._make_layer(res_block, 4, 128, stride=2)
        self.layer3 = self._make_layer(res_block, 6, 256, stride=2)
        self.layer4 = self._make_layer(res_block, 3, 512, stride=2)

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * self.expansion, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
    def _make_layer(self, block, num, out_channels, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding = 0, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * self.expansion
        for _ in range (num - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
class CNN_plus_Transformer(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()
        self.cnn = CNN(num_classes=17)
        self.cnn_to_3 = nn.Conv2d(512*4, 3, kernel_size=1)
        self.mobilevit = timm.create_model('mobilevit_s', pretrained=True, features_only = True)
        channels = self.mobilevit.feature_info.channels()[-1]  # Get the number of channels at each stage
        self.head = nn.Sequential(
            nn.Conv2d(channels, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 17, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.cnn(x)
        mapped_feat = self.cnn_to_3(out)
        feats = self.mobilevit(mapped_feat)[-1]
        heatmap = self.head(feats)
        heatmap = F.interpolate(heatmap, size=(64,64), mode='bilinear', align_corners=False)
        return heatmap