import torch
import torch.nn as nn
import math
import copy


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MoboStages(nn.Module):
    def __init__(self, block, interverted_residual_setting, width_mult, input_channel, compress=None):
        super(MoboStages, self).__init__()
        self.layers = []

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.layers.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.layers.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.layers = nn.Sequential(*self.layers)
        self.compress = copy.deepcopy(compress)

        if type(self.compress) is list or type(self.compress) is tuple:
            for idx, module in enumerate(self.compress):
                self.add_module("compress" + str(idx), mtaodule)

    @staticmethod
    def _to_cpu(x):
        if type(x) is list:
            x_cpu_list = []
            x_list = x
            for x in x_list:
                x_cpu_list.append(MoboStages._to_cpu(x))
            return x_cpu_list
        elif type(x) is tuple:
            x_cpu_list = []
            x_tuple = x
            for x in x_tuple:
                x_cpu_list.append(MoboStages._to_cpu(x))
            return tuple(x_cpu_list)
        else:
            return x.cpu()

    def compress_replace(self, compress_new):
        """
        If replace compress method, beware initialization of parameters in new compress method.
        A compress method must be pair of encoder and decoder
        Args:
             compress_new:  compress_new can be a tuple of encoder-decoder pair, or tuple of
            encoder list and decoder list
        """
        self.compress = copy.deepcopy(compress_new)
        if type(self.compress) is list or type(self.compress) is tuple:
            for idx, module in enumerate(self.compress):
                self.add_module("compress" + str(idx), module)

    def update(self):
        if type(self.compress) is list or type(self.compress) is tuple:
            for indx, block in enumerate(self.layers):
                self.compress[indx].update()
        else:
            self.compress.update()

    def forward(self, x):
        feature_maps = []  # TODO clean up
        fm_transforms = []  # TODO clean up
        for indx, block in enumerate(self.layers):
            x = block(x)
            feature_maps.append(x.cpu())
            if self.compress is not None:
                if type(self.compress) is list or type(self.compress) is tuple:
                    x_re, fm_transform = self.compress[indx](x)
                    fm_transforms.append(self._to_cpu(fm_transform))
                else:
                    x_re, fm_transform = self.compress(x.detach())
                    fm_transforms.append(self._to_cpu(fm_transform))
                x = x_re

        return x, feature_maps, fm_transforms


class MobileNetV2Cifar(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2Cifar, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.preStage = [conv_bn(3, input_channel, 2)]
        self.preStage = nn.Sequential(*self.preStage)

        # building inverted residual blocks
        self.stages = MoboStages(block, interverted_residual_setting, width_mult, input_channel)

        # building last several layers
        self.postStage = []
        self.postStage.append(conv_1x1_bn(320, self.last_channel))
        self.postStage = nn.Sequential(*self.postStage)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def _to_cpu(x):
        if type(x) is list:
            x_cpu_list = []
            x_list = x
            for x in x_list:
                x_cpu_list.append(MoboStages._to_cpu(x))
            return x_cpu_list
        elif type(x) is tuple:
            x_cpu_list = []
            x_tuple = x
            for x in x_tuple:
                x_cpu_list.append(MoboStages._to_cpu(x))
            return tuple(x_cpu_list)
        else:
            return x.cpu()

    def forward(self, x):
        feature_maps = []  # TODO clean up
        fm_transforms = []  # TODO clean up

        assert not torch.sum(torch.isnan(x)), x.size()
        x = self.preStage(x)
        x, feature_maps, fm_transform = self.stages(x)
        x = self.postStage(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x, feature_maps, fm_transform

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # TODO generalize
    def compress_replace(self, compress_new):
        self.stages.compress_replace(compress_new)

    def update(self):
        self.stages.update()
