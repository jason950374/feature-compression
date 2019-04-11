import torch
import torch.nn as nn
import torch.nn.functional as F
import functional.dct as dct


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
     :param in_planes: Input channel number
     :param out_planes: Out channel number
     :param stride: Stride of the convolution. Default: 1
     :return: nn.Conv2d instance"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    """
    BasicBlock module for ResNet
    conv -> bn -> relu -> conv -> bn
    residual connection is NOT included yet
    :param inplanes: Input channel number
    :param planes: Out channel number
    :param stride: Stride of the convolution. Default: 1"""
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    """
    Bottleneck module for ResNet
    conv -> bn -> relu -> conv -> bn
    residual connection is NOT included yet
    :param inplanes: input channel number
    :param planes: out channel number
    :param stride: Stride of the convolution. Default: 1"""
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class ResNetBlock(nn.Module):
    """
            ResNet_block module for ResNet
            add residual connection on given block
            :param block: block module instance, BasicBlock or Bottleneck for instance
            :param downsample: downsample module instance"""
    def __init__(self, block, downsample=None):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        self.block = block

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNetStages(nn.Module):
    """
    Module  of ResNet in mid stage
    :param block: Block type for ResNet, a nn.Module class
     :param ns: Number of blocks in each downsampling stages, a list or tuple
     :param in_planes: Input channel number for first stage"""
    def __init__(self, block, ns, in_planes):
        super(ResNetStages, self).__init__()
        self.layers = []
        self.in_planes = in_planes
        first_stage = True
        planes_cur = self.in_planes
        in_planes_cur = self.in_planes
        for block_num in ns:
            layer, in_planes_cur \
                = self._make_layer(block, in_planes_cur, planes_cur, block_num, stride=1 if first_stage else 2)
            self.layers += [*layer]
            first_stage = False
            planes_cur *= 2

        self.layers = nn.Sequential(*self.layers)

    @staticmethod
    def _make_layer(block, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or block.expansion != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [ResNetBlock(block(in_planes, planes, stride), downsample)]
        in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(ResNetBlock(block(in_planes, planes)))

        return nn.Sequential(*layers), in_planes

    def forward(self, x):
        feature_maps = []  # TODO clean up
        feature_maps_dct = []  # TODO clean up
        for indx, block in enumerate(self.layers):
            x = block(x)
            # scale = 32
            # x = torch.round(torch.clamp(x * scale, 0, 255)) / scale
            feature_maps.append(x.cpu())
            X = dct.dct_2d(x)  # TODO clean up and BP path

            prune_size = x.size(-1) // 2
            for i in range(x.size(-1)):
                if i < prune_size:
                    X[:, :, :-(prune_size - i), -(i + 1)] = torch.clamp(torch.round(X[:, :, :-(prune_size - i), -(i + 1)] / 4), -128, 127)
                    X[:, :, -(prune_size - i):, -(i + 1)] = torch.clamp(torch.round(X[:, :, -(prune_size - i):, -(i + 1)] / 4), -128, 127)
                else:
                    X[:, :, :, -(i + 1)] = torch.clamp(torch.round(X[:, :, :, -(i + 1)] / 4), -128, 127)
            x = dct.idct_2d(X * 4)

            feature_maps_dct.append(X.cpu())
        return x, feature_maps, feature_maps_dct


class ResNetCifar(nn.Module):
    """
    Module  of ResNet for Cifar dataset
    Always use BasicBlock
    channels  of each stage16, 32, 64
    :param depth: Depth of block
    :param num_classes: Number of classes, Default: 10 for Cifar-10"""
    def __init__(self, depth, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        block = BasicBlock
        ns = (depth - 2) // 6
        ns = [ns] * 3
        self.stages = ResNetStages(block, ns, 16)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, feature_maps, feature_maps_dct = self.stages(x)  # TODO clean up
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feature_maps, feature_maps_dct
