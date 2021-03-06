import copy
import torch.nn as nn
import torch.nn.functional as F
from model.compress import Compress


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
     :param in_planes: Input channel number
     :param out_planes: Out channel number
     :param stride: Stride of the convolution. Default: 1
     :return: nn.Conv2d instance"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def identical(x):
    return x, x


class BasicBlock(nn.Module):
    """
    BasicBlock module for ResNet
    conv -> bn -> relu -> conv -> bn
    residual connection is NOT included yet
    
    Args:
        inplanes: Input channel number
        planes: Out channel number
        stride: Stride of the convolution. Default: 1
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, compress=identical):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.compress = compress

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out, fm_transform = self.compress(out)
        feature_map = out
        out = self.conv2(out)
        out = self.bn2(out)

        return out, feature_map, fm_transform

    def compress_replace_inblock(self, compress_new):
        self.compress = compress_new

    def update(self):
        if isinstance(self.compress, Compress):
            self.compress.update()


class Bottleneck(nn.Module):
    """
    Bottleneck module for ResNet
    conv -> bn -> relu -> conv -> bn
    residual connection is NOT included yet

    Args:
        inplanes: input channel number
        planes: out channel number
        stride: Stride of the convolution. Default: 1
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, compress=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.compress1 = compress[0]
        self.compress2 = compress[1]

    def forward(self, x):
        feature_maps = []
        fm_transforms = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out, fm_transform = self.compress1(out)
        feature_maps.append(out)
        fm_transforms.append(fm_transform)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out, fm_transform = self.compress1(out)
        feature_maps.append(out)
        fm_transforms.append(fm_transform)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

    def compress_replace_inblock(self, compress_new):
        self.compress1 = compress_new[0]
        self.compress2 = compress_new[1]


class ResNetBlock(nn.Module):
    """
    ResNet_block module for ResNet
    add residual connection on given block

    Args:
        block: block module instance, BasicBlock or Bottleneck for instance
        downsample: downsample module instance
    """
    def __init__(self, block, downsample=None):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        self.block = block

    def forward(self, x):
        identity = x
        out, feature_map_block, fm_transform = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)

        return out, feature_map_block, fm_transform

    def compress_replace_inblock(self, compress_new):
        self.block.compress_replace_inblock(compress_new)

    def update(self):
        self.block.update()


class ResNetStages(nn.Module):
    """
    Module  of ResNet in mid stage

    Args:
        block: Block type for ResNet, a nn.Module class
        ns: Number of blocks in each downsampling stages, a list or tuple
        in_planes: Input channel number for first stage
        compress: a module or function to compress feature map
    """
    def __init__(self, block, ns, in_planes, compress=identical):
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

        self.compress = copy.deepcopy(compress)
        if type(self.compress) is list or type(self.compress) is tuple:
            self.compress = nn.ModuleList(self.compress)

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

    @staticmethod
    def _to_cpu(x):
        if type(x) is list:
            x_cpu_list = []
            x_list = x
            for x in x_list:
                x_cpu_list.append(ResNetStages._to_cpu(x))
            return x_cpu_list
        elif type(x) is tuple:
            x_cpu_list = []
            x_tuple = x
            for x in x_tuple:
                x_cpu_list.append(ResNetStages._to_cpu(x))
            return tuple(x_cpu_list)
        else:
            return x.cpu()

    def compress_replace_branch(self, compress_new):
        """
        If replace compress method, beware initialization of parameters in new compress method.
        A compress method must be pair of encoder and decoder
        Args:
             compress_new:  compress_new can be a tuple of encoder-decoder pair, or tuple of
            encoder list and decoder list
        """
        self.compress = copy.deepcopy(compress_new)
        if type(self.compress) is list or type(self.compress) is tuple:
            self.compress = nn.ModuleList(self.compress)

    def compress_replace_inblock(self, compress_new):
        for idx, block in enumerate(self.layers):
            if type(compress_new) is list or type(compress_new) is tuple:
                block.compress_replace_inblock(compress_new[idx])
            else:
                block.compress_replace_inblock(copy.deepcopy(compress_new))

    def update(self):
        if type(self.compress) is nn.ModuleList:
            for indx, block in enumerate(self.layers):
                self.compress[indx].update()
                block.update()
        elif isinstance(self.compress, Compress):
            self.compress.update()

    def forward(self, x):
        feature_maps_branch = []  # TODO clean up
        fm_transforms_branch = []   # TODO clean up
        fm_transforms_block = []
        feature_maps_block = []
        for block, compress in zip(self.layers, self.compress):
            x, feature_map_block, fm_transform_block = block(x)
            x_re, fm_transform = compress(x)

            feature_maps_block.append(feature_map_block)
            fm_transforms_block.append(fm_transform_block)
            feature_maps_branch.append(x)
            fm_transforms_branch.append(fm_transform)

            x = x_re

        return x, feature_maps_branch, fm_transforms_branch, feature_maps_block, fm_transforms_block


class ResNetCifar(nn.Module):
    """
    Module  of ResNet for Cifar dataset
    Always use BasicBlock
    channels  of each stage: 16, 32, 64
    Args:
        depth (int): Depth of block
        num_classes (int): Number of classes, Default: 10 for Cifar-10"""
    def __init__(self, depth, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
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
        x = F.relu(x)
        # TODO clean up
        x, feature_maps_branch, fm_transforms_branch, feature_maps_block, fm_transforms_block = self.stages(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feature_maps_branch, fm_transforms_branch, feature_maps_block, fm_transforms_block

    def compress_replace_branch(self, compress_new):
        self.stages.compress_replace_branch(compress_new)

    def compress_replace_inblock(self, compress_new):
        self.stages.compress_replace_inblock(compress_new)

    def update(self):
        self.stages.update()


class ResNetImageNet(nn.Module):
    """
        Module  of ResNet for ImageNet dataset
        channels  of each stage: 64, 128, 256, 512

        Args:
            block (nn.Module): Type of block used
            layers (list): List of blocks amount for each states
            zero_init_residual (bool): Zero-initialize the last BN in each residual branch,
                so that the residual branch starts with zeros, and each residual block behaves like an identity.
                This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677"""

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNetImageNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = ResNetStages(block, layers, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # TODO clean up
        x, feature_maps_branch, fm_transforms_branch, feature_maps_block, fm_transforms_block = self.stages(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feature_maps_branch, fm_transforms_branch, feature_maps_block, fm_transforms_block

    def compress_replace_branch(self, compress_new):
        self.stages.compress_replace_branch(compress_new)

    def compress_replace_inblock(self, compress_new):
        self.stages.compress_replace_inblock(compress_new)

    def update(self):
        self.stages.update()


def resnet18(zero_init_residual=False):
    """Constructs a ResNet-18 model.

    Args:
        zero_init_residual (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetImageNet(BasicBlock, [2, 2, 2, 2], zero_init_residual)

    return model
