import math
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['iresnet18_hdfg_arcface', 'iresnet34_hdfg_arcface', 'iresnet50_hdfg_arcface', 'iresnet100_hdfg_arcface', 'iresnet200_hdfg_arcface']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class ArcFace(nn.Module):
    def __init__(self, num_classes, embed_dim=256, margin=0.4, scale=32, easy_margin=True):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.weights = nn.Parameter(torch.Tensor(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weights)

    def forward(self, features, labels):
        # 归一化特征和权重
        features = F.normalize(features, p=2, dim=1)
        weights = F.normalize(self.weights, p=2, dim=1)

        # 正确计算目标类别的cos(theta + margin)
        cosine = F.linear(features, weights)

        # 如果labels为None，直接返回cosine作为logits（用于FLOPs计算等场景）
        if labels is None:
            return cosine

        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # 仅修改目标类别的theta
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        target_theta = theta + self.margin * one_hot

        if self.easy_margin:
            target_theta = torch.where(theta > math.pi-self.margin, theta-self.margin, target_theta)
        logits = torch.cos(target_theta)
        logits = self.scale * (one_hot * logits + (1.0 - one_hot) * cosine)
        return logits


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, is_last=False):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.is_last = is_last
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.is_last:
            return out, out  # pre-activation feature for distillation
        return out

    def forward(self, x):
        return self.forward_impl(x)


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
    def __init__(self, channel, ratio=2, block_name='IBasicBlock', reduction=16):
        super(HDFG, self).__init__()
        self.inplanes = channel
        self.fg = FeatGeneration(self.inplanes, ratio=ratio, dw_size=3, bias=False, relu=True)
        self.se = SEModule(self.inplanes * ratio, reduction=reduction)
        self.conv1 = nn.Conv2d(self.inplanes * ratio, self.inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)

    def forward(self, feat):
        expanded_feat = self.fg(feat)
        attended_feat = self.se(expanded_feat)
        out = self.conv1(attended_feat)
        out = self.bn(out)
        return out, expanded_feat, attended_feat


class IResNet_HDFG(nn.Module):
    def __init__(self,
                 block, layers, dropout=0.0, num_features=512, m=0.5, s=64, num_classes=10575, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False,
                 ratio=2, sbkd_feature='post_attention'):
        super(IResNet_HDFG, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        if sbkd_feature not in ('pre_attention', 'post_attention'):
            raise ValueError("sbkd_feature must be 'pre_attention' or 'post_attention'")
        self.sbkd_feature = sbkd_feature
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.hdfg1 = HDFG(64 * block.expansion, ratio, 'IBasicBlock', reduction=16)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.hdfg2 = HDFG(128 * block.expansion, ratio, 'IBasicBlock', reduction=16)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.hdfg3 = HDFG(256 * block.expansion, ratio, 'IBasicBlock', reduction=16)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.hdfg4 = HDFG(512 * block.expansion, ratio, 'IBasicBlock', reduction=16)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        # self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.arcface = ArcFace(num_classes, num_features, margin=m, scale=s)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, labels, is_feat=False):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x, f1_pre, f1_post = self.hdfg1(x)
            f1 = f1_pre if self.sbkd_feature == 'pre_attention' else f1_post
            x = self.layer2(x)
            x, f2_pre, f2_post = self.hdfg2(x)
            f2 = f2_pre if self.sbkd_feature == 'pre_attention' else f2_post
            x = self.layer3(x)
            x, f3_pre, f3_post = self.hdfg3(x)
            f3 = f3_pre if self.sbkd_feature == 'pre_attention' else f3_post
            x = self.layer4(x)
            x, f4_pre, f4_post = self.hdfg4(x)
            f4 = f4_pre if self.sbkd_feature == 'pre_attention' else f4_post
            x = self.bn2(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        x = self.arcface(x, labels)
        if is_feat:
            return [f1, f2, f3, f4], x
        return x


class Auxiliary_Branch(nn.Module):
    """辅助分支: 对每个 HDFG 中间特征做 SE + 子网络提取 + ArcFace 分类
    参照 scr/resnet_HDFG_ArcFace.py: 分支使用 ArcFace 头增强监督信号
    """
    def __init__(self, block, layers, num_classes, ratio=2, embed_dim=256):
        super(Auxiliary_Branch, self).__init__()
        self.embed_dim = embed_dim
        self.inplanes = int(64 * block.expansion * ratio)
        self.block_extractor1 = nn.Sequential(*[
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2)])
        self.inplanes = int(128 * block.expansion * ratio)
        self.block_extractor2 = nn.Sequential(*[
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2)])
        self.inplanes = int(256 * block.expansion * ratio)
        self.block_extractor3 = nn.Sequential(*[
            self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = int(512 * block.expansion * ratio)
        self.block_extractor4 = nn.Sequential(*[
            self._make_layer(block, 512, layers[3], stride=1)])  # 不下采样, 保留空间信息

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # ArcFace 头 (参考 scr: fc → embed_dim → ArcFace)
        self.fc1 = nn.Linear(512 * block.expansion, embed_dim)
        self.fc2 = nn.Linear(512 * block.expansion, embed_dim)
        self.fc3 = nn.Linear(512 * block.expansion, embed_dim)
        self.fc4 = nn.Linear(512 * block.expansion, embed_dim)
        self.arcface1 = ArcFace(num_classes, embed_dim, margin=0.2, scale=16)
        self.arcface2 = ArcFace(num_classes, embed_dim, margin=0.2, scale=16)
        self.arcface3 = ArcFace(num_classes, embed_dim, margin=0.2, scale=16)
        self.arcface4 = ArcFace(num_classes, embed_dim, margin=0.2, scale=16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=False))
        return nn.Sequential(*layers)

    def forward(self, x, labels):
        """x: [f1,f2,f3,f4] HDFG中间特征列表, labels: 类别标签"""
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor' + str(idx))(x[i])
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = getattr(self, 'fc' + str(idx))(out)
            out = getattr(self, 'arcface' + str(idx))(out, labels)
            ss_logits.append(out)
        return ss_logits


class IResNet_HDFG_Aux(nn.Module):
    def __init__(self, block, layers, num_classes, is_aux=False, dropout=0.1, ratio=2, **kwargs):
        super(IResNet_HDFG_Aux, self).__init__()
        self.is_aux = is_aux
        self.backbone = IResNet_HDFG(block, layers, dropout=dropout, num_classes=num_classes, ratio=ratio, **kwargs)
        if self.is_aux:
            self.auxiliary_branch = Auxiliary_Branch(block, layers, num_classes, ratio)

    def forward(self, x, labels=None, is_feat=False, grad=False, is_aux=False):
        if is_feat:
            mid_feats, logits = self.backbone(x, labels, is_feat=is_feat)
            if grad is False:
                mid_feats = [feat.detach() for feat in mid_feats]
            if is_aux:
                ss_logits = self.auxiliary_branch(mid_feats, labels)
                return logits, ss_logits
            return mid_feats, logits
        else:
            logits = self.backbone(x, labels, is_feat=is_feat)
            return logits


def iresnet18_hdfg_arcface(num_classes=100, is_aux=False, dropout=0.1, ratio=2, **kwargs):
    return IResNet_HDFG_Aux(IBasicBlock, [2, 2, 2, 2], num_classes=num_classes, is_aux=is_aux, dropout=dropout, ratio=ratio, **kwargs)


def iresnet34_hdfg_arcface(num_classes=100, is_aux=False, dropout=0.1, ratio=2, **kwargs):
    return IResNet_HDFG_Aux(IBasicBlock, [3, 4, 6, 3], num_classes=num_classes, is_aux=is_aux, dropout=dropout, ratio=ratio,  **kwargs)


def iresnet50_hdfg_arcface(num_classes=100, is_aux=False, dropout=0.1, ratio=2, **kwargs):
    return IResNet_HDFG_Aux(IBasicBlock, [3, 4, 14, 3], num_classes=num_classes, is_aux=is_aux, dropout=dropout, ratio=ratio, **kwargs)


def iresnet100_hdfg_arcface(num_classes=100, is_aux=False, dropout=0.1, ratio=2, **kwargs):
    return IResNet_HDFG_Aux(IBasicBlock, [3, 13, 30, 3], num_classes=num_classes, is_aux=is_aux, dropout=dropout, ratio=ratio, **kwargs)


def iresnet200_hdfg_arcface(num_classes=100, is_aux=False, dropout=0.1, ratio=2, **kwargs):
    return IResNet_HDFG_Aux(IBasicBlock, [6, 26, 60, 6], num_classes=num_classes, is_aux=is_aux, dropout=dropout, ratio=ratio, **kwargs)
