import copy
import torch.nn as nn
import torch.nn.functional as F


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
    :param in_planes: Input channel number for first stage
    :param  compress: a module or function to compress feature map
    """
    def __init__(self, block, ns, in_planes, compress=None):
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
        self.compress = compress

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

    def compress_replace(self, compress_new):
        """
            If replace compress method, beware initialization of parameters in new compress method
            if compress_new is nn.Module, will perform deepcopy
            :param compress_new:  new compress module or function
            """
        if isinstance(compress_new, nn.Module):
            self.compress = copy.deepcopy(compress_new)
        else:
            self.compress = compress_new

    def forward(self, x):
        feature_maps = []  # TODO clean up
        for indx, block in enumerate(self.layers):
            x = block(x)
            feature_maps.append(x.cpu())
            if self.compress is not None:
                x = self.compress(x)

        return x, feature_maps


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
        x, feature_maps = self.stages(x)  # TODO clean up
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feature_maps

    # TODO generalize
    def compress_replace(self, compress_new):
        self.stages.compress_replace(compress_new)
