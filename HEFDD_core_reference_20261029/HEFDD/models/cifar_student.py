from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet56_HDFG', 'resnet20_HDFG', 'resnet32x4_HDFG', 'resnet8x4_HDFG', 'resnet8_HDFG', 'resnet8x4_HDFG']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeatGeneration(nn.Module):
    def __init__(self, inplanes, ratio=2, dw_size=3, bias=False, relu=True):
        super(FeatGeneration, self).__init__()
        self.inplanes = inplanes
        new_channels = inplanes * (ratio - 1)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.inplanes, new_channels, dw_size, stride=1, padding=dw_size // 2, groups=self.inplanes,
                      bias=bias),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.cheap_operation(x)
        out = torch.cat([x, x1], dim=1)
        return out


class HDFG(nn.Module):
    def __init__(self, channel, ratio=2, block_name='BasicBlock', reduction=16):
        super(HDFG, self).__init__()
        if block_name.lower() == 'basicblock':
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
        self.inplanes = channel
        self.fg = FeatGeneration(self.inplanes, ratio=ratio, dw_size=3, bias=False, relu=True)
        self.se = SEModule(self.inplanes * ratio, reduction=reduction)
        self.conv1 = nn.Conv2d(self.inplanes * ratio, self.inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)

    def forward(self, feat):
        out = self.fg(feat)
        mid_feat = self.se(out)
        out = self.conv1(mid_feat)
        out = self.bn(out)
        return out, mid_feat


class ResNet_HDFG(nn.Module):
    def __init__(self, depth, num_filters, block_name='BasicBlock', ratio=2, reduction=16, num_classes=100):
        super(ResNet_HDFG, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.fg1 = HDFG(num_filters[1] * block.expansion, ratio, block_name, reduction=reduction)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.fg2 = HDFG(num_filters[2] * block.expansion, ratio, block_name, reduction=reduction)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.fg3 = HDFG(num_filters[3] * block.expansion, ratio, block_name, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        out, f1 = self.fg1(x)
        x = self.layer2(out)
        out, f2 = self.fg2(x)
        x = self.layer3(out)
        out, f3 = self.fg3(x)
        #print('f3', f3.size())
        x = self.avgpool(out)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)

        return [f1, f2, f3], logit



class Auxiliary_Branch(nn.Module):
    def __init__(self, depth, num_filters, ratio=2, block_name='BasicBlock', num_classes=100):
        super(Auxiliary_Branch, self).__init__()
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[1] * block.expansion * ratio
        self.block_extractor1 = nn.Sequential(*[self._make_layer(block, num_filters[2], n, stride=2),
                                                self._make_layer(block, num_filters[3], n, stride=2)])
        self.inplanes = num_filters[2] * block.expansion * ratio
        self.block_extractor2 = nn.Sequential(*[self._make_layer(block, num_filters[3], n, stride=2)])
        self.inplanes = num_filters[3] * block.expansion * ratio
        self.block_extractor3 = nn.Sequential(*[self._make_layer(block, num_filters[3], n, stride=1)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.fc2 = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.fc3 = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = getattr(self, 'fc' + str(idx))(out)
            ss_logits.append(out)

        return ss_logits


class ResNet_Auxiliary(nn.Module):
    def __init__(self, depth, num_filters, block_name='BasicBlock', ratio=2, reduction=16, num_classes=100):
        super(ResNet_Auxiliary, self).__init__()
        self.backbone = ResNet_HDFG(depth, num_filters, block_name, ratio=ratio, reduction=reduction, num_classes=num_classes)
        self.auxiliary_branch = Auxiliary_Branch(depth, num_filters, ratio, block_name, num_classes)

    def forward(self, x, grad=False, kd=True):
        feats, logit = self.backbone(x)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
                # print(feats[i].shape)
        if kd is False:
            return logit
        else:
            ss_logits = self.auxiliary_branch(feats)
            return logit, ss_logits


def resnet8_HDFG(**kwargs):
    return ResNet_Auxiliary(8, [16, 16, 32, 64], 'basicblock', **kwargs)



def resnet14_HDFG(**kwargs):
    return ResNet_Auxiliary(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20_HDFG(**kwargs):
    return ResNet_Auxiliary(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20_aux_HDFG(**kwargs):
    return ResNet_Auxiliary(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet14x05_HDFG(**kwargs):
    return ResNet_Auxiliary(14, [8, 8, 16, 32], 'basicblock', **kwargs)


def resnet20x05_HDFG(**kwargs):
    return ResNet_Auxiliary(20, [8, 8, 16, 32], 'basicblock', **kwargs)


def resnet20x0375_HDFG(**kwargs):
    return ResNet_Auxiliary(20, [6, 6, 12, 24], 'basicblock', **kwargs)


def resnet32_HDFG(**kwargs):
    return ResNet_Auxiliary(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet44_HDFG(**kwargs):
    return ResNet_Auxiliary(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet56_HDFG(**kwargs):
    return ResNet_Auxiliary(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet110_HDFG(**kwargs):
    return ResNet_Auxiliary(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet8x4_HDFG(**kwargs):
    return ResNet_Auxiliary(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def resnet32x4_HDFG(**kwargs):
    return ResNet_Auxiliary(32, [32, 64, 128, 256], 'basicblock', **kwargs)


