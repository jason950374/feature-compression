import os.path as osp
import copy
import fcn
import torch.nn as nn

from .fcn32s import get_upsampling_weight


class FCN8s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
            path=cls.pretrained_model,
            md5='dbd9bbb3829a3184913bccc74373afbb',
        )

    def __init__(self, n_class=21, compress=None):
        super(FCN8s, self).__init__()
        self.compress = copy.deepcopy(compress)

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        feature_maps = []
        fm_transforms = []
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 0)
        h = self.relu1_2(self.conv1_2(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 1)
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 2)
        h = self.relu2_2(self.conv2_2(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 3)
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 4)
        h = self.relu3_2(self.conv3_2(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 5)
        h = self.relu3_3(self.conv3_3(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 6)
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 7)
        h = self.relu4_2(self.conv4_2(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 8)
        h = self.relu4_3(self.conv4_3(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 9)
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 10)
        h = self.relu5_2(self.conv5_2(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 11)
        h = self.relu5_3(self.conv5_3(h))
        h = self._apply_compress(h, feature_maps, fm_transforms, 12)
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h, feature_maps, fm_transforms

    def _apply_compress(self, x, feature_maps, fm_transforms, indx):
        feature_maps.append(x)
        if self.compress is not None:
            if type(self.compress) is list or type(self.compress) is tuple:
                x_re, fm_transform = self.compress[indx](x)
                fm_transforms.append(fm_transform)
            else:
                x_re, fm_transform = self.compress(x)
                fm_transforms.append(fm_transform)
            return x_re
        else:
            return x

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

    def compress_replace(self, compress_new):
        self.compress = copy.deepcopy(compress_new)
        if type(self.compress) is list or type(self.compress) is tuple:
            for idx, module in enumerate(self.compress):
                self.add_module("compress" + str(idx), module)


class FCN8sAtOnce(FCN8s):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    def forward(self, x):
        feature_maps = []
        h = x
        h = self.relu1_1(self.conv1_1(h))
        feature_maps.append(h)
        _, _, = self.compress[0](h)
        h = self.relu1_2(self.conv1_2(h))
        feature_maps.append(h)
        _, _, = self.compress[1](h)
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        feature_maps.append(h)
        _, _, = self.compress[2](h)
        h = self.relu2_2(self.conv2_2(h))
        feature_maps.append(h)
        _, _, = self.compress[3](h)
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        feature_maps.append(h)
        _, _, = self.compress[4](h)
        h = self.relu3_2(self.conv3_2(h))
        feature_maps.append(h)
        _, _, = self.compress[5](h)
        h = self.relu3_3(self.conv3_3(h))
        feature_maps.append(h)
        _, _, = self.compress[6](h)
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        feature_maps.append(h)
        _, _, = self.compress[7](h)
        h = self.relu4_2(self.conv4_2(h))
        feature_maps.append(h)
        _, _, = self.compress[8](h)
        h = self.relu4_3(self.conv4_3(h))
        feature_maps.append(h)
        _, _, = self.compress[9](h)
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        feature_maps.append(h)
        _, _, = self.compress[10](h)
        h = self.relu5_2(self.conv5_2(h))
        feature_maps.append(h)
        _, _, = self.compress[11](h)
        h = self.relu5_3(self.conv5_3(h))
        feature_maps.append(h)
        _, _, = self.compress[12](h)
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h, feature_maps

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
