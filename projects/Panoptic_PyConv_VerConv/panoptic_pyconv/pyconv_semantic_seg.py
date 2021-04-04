import torch
from torch import nn
import torch.nn.functional as F
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from typing import Dict
import numpy as np
import fvcore.nn.weight_init as weight_init

from detectron2.layers import (
    ShapeSpec,
    get_norm,
)


class PyConv2d(nn.Module):
    """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``

    Example::

        >>> # PyConv with two pyramid levels, kernels: 3x3, 5x5
        >>> m = PyConv2d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)

        >>> # PyConv with three pyramid levels, kernels: 3x3, 5x5, 7x7
        >>> m = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        >>> input = torch.randn(4, 64, 56, 56)
        >>> output = m(input)
    """

    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()

        # if stride = 1, the  spatial size is preserved.
        self.conv2_1 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[0], stride=stride,
                                 padding=pyconv_kernels[0] // 2, dilation=1, groups=pyconv_groups[0], bias=False)
        self.conv2_2 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[1], stride=stride,
                                 padding=pyconv_kernels[1] // 2, dilation=1, groups=pyconv_groups[1], bias=False)
        self.conv2_3 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[2], stride=stride,
                                 padding=pyconv_kernels[2] // 2, dilation=1, groups=pyconv_groups[2], bias=False)
        self.conv2_4 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[3], stride=stride,
                                 padding=pyconv_kernels[3] // 2, dilation=1, groups=pyconv_groups[3], bias=False)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class GlobalPyConvBlock(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, norm):
        super(GlobalPyConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(bins),  # reduce spatial size but retain the channel size
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            get_norm(norm, reduction_dim),
            # BatchNorm(reduction_dim),
            nn.ReLU(inplace=True),
            PyConv4(reduction_dim, reduction_dim),
            get_norm(norm, reduction_dim),
            # BatchNorm(reduction_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_dim, reduction_dim, kernel_size=1, bias=False),
            get_norm(norm, reduction_dim),
            # BatchNorm(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        x = F.interpolate(self.features(x), x_size[2:], mode='bilinear',
                          align_corners=True)  # back to size before AdpAvgPool
        return x


class LocalPyConvBlock(nn.Module):
    def __init__(self, inplanes, planes, norm, reduction1=4):
        super(LocalPyConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            get_norm(norm, inplanes // reduction1),
            nn.ReLU(inplace=True),
            PyConv4(inplanes // reduction1, inplanes // reduction1),
            get_norm(norm, inplanes // reduction1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction1, planes, kernel_size=1, bias=False),
            get_norm(norm, planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.layers(x)  # spatial size is retained


class MergeLocalGlobal(nn.Module):
    def __init__(self, inplanes, planes, norm):
        super(MergeLocalGlobal, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, groups=1, bias=False),  # preserves the spatial size
            get_norm(norm, planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x  # spatial size is preserves


class PyConvHead(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 norm,
                 out_size_local_context,
                 out_size_global_context,
                 reduction_local=4,
                 bins=9):
        """
        Except inplanes, all parameters are hyper-parameters.
        Args:
            inplanes:
            planes:
            BatchNorm:

        Remarks:
        We need to add upsampling and loss to the head.
        """
        super(PyConvHead, self).__init__()

        # The following two parameters can be set to hyper-parameters as well.
        # out_size_local_context = 512
        # out_size_global_context = 512

        # By setting strides to be 1, LocalPyConvBlock and GlobalPyConvBlock  preserve the spatial size
        # self.local_context = LocalPyConvBlock(inplanes, out_size_local_context, norm, reduction1=4)
        self.local_context = LocalPyConvBlock(inplanes, out_size_local_context, norm, reduction1=reduction_local)
        # self.global_context = GlobalPyConvBlock(inplanes, out_size_global_context, 9, norm)
        self.global_context = GlobalPyConvBlock(inplanes, out_size_global_context, bins, norm)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_context, planes, norm)

    def forward(self, x):
        x = self.merge_context(self.local_context(x), self.global_context(x))
        return x


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegMultiScalePyConvHead(nn.Module):
    """
    A semantic segmentation head we call MultiScalePyConvHead.
    It takes FPN features as input and merges info from all levels
    of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES  # P2, P3, P4, ....
        feature_strides = {k: v.stride for k, v in input_shape.items()}  # Pi: stride
        feature_channels = {k: v.channels for k, v in input_shape.items()}  # Pi: channel
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        out_channels_local = cfg.MODEL.SEM_SEG_HEAD.PYCONV_OUT_SIZE_LOCAL_CONTEXT
        out_channels_global = cfg.MODEL.SEM_SEG_HEAD.PYCONV_OUT_SIZE_GLOBAL_CONTEXT
        out_channels_merge = cfg.MODEL.SEM_SEG_HEAD.PYCONV_MERGE_OUT_SIZE
        local_reduction = cfg.MODEL.SEM_SEG_HEAD.PYCONV_LOCAL_REDUCTION
        global_bins = cfg.MODEL.SEM_SEG_HEAD.PYCONV_GLOBAL_BINS
        cls_dropout = cfg.MODEL.SEM_SEG_HEAD.PYCONV_CLS_DROPOUT
        self.fuse_mode = cfg.MODEL.SEM_SEG_HEAD.PYCONV_FUSE_MODE

        # define the modules for all scales of choice
        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = [PyConvHead(feature_channels[in_feature],
                                   out_channels_merge,
                                   norm,
                                   out_channels_local,
                                   out_channels_global,
                                   local_reduction,
                                   global_bins),
                        nn.Upsample(scale_factor=feature_strides[in_feature],
                                    mode="bilinear",
                                    align_corners=False)
                        ]
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        if self.fuse_mode == "SUM":
            # Add up outputs of PyConvSemSegHeads from different FPN levels.
            pred_in_channels = out_channels_merge
        else:
            # Concatenate outputs of PyConvSemSegHeads from different FPN levels along channel dimension.
            pred_in_channels = out_channels_merge * len(self.in_features)
        self.predictor = nn.Sequential(
            nn.Dropout2d(p=cls_dropout),
            nn.Conv2d(pred_in_channels, num_classes, kernel_size=1)
        )  # nn.Sequential calls the nn.add_module()

    def forward(self, features, targets=None):
        """
        features: {p2:.., p3:.., p4:...,.....}
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (Bxnum_classxHxW logits, {})
        """
        predictions = self.layers(features)  # (N, num_classes, H, W)
        if self.training:
            return None, self.losses(predictions, targets)
        else:
            return predictions, {}

    def layers(self, features):
        if self.fuse_mode == "SUM":
            x = None
            for i, f in enumerate(self.in_features):
                if i == 0:
                    x = self.scale_heads[i](features[f])
                else:
                    x += self.scale_heads[i](features[f])
        else:
            # concatenate features from levels.
            outs = []
            for i, f in enumerate(self.in_features):
                outs.append(self.scale_heads[i](features[f]))  # (N, out_channels_merge, H, W)
            x = torch.cat(outs, dim=1)  # (N, out_channels_merge * len(features), H, W)
        x = self.predictor(x)  # (N, num_classes, H, W)
        return x

    def losses(self, predictions, targets):
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegPyConvHeadV1(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES  # p2, p3, p4, ....
        feature_strides = {k: v.stride for k, v in input_shape.items()}  # pi: stride
        feature_channels = {k: v.channels for k, v in input_shape.items()}  # pi: channel
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride = 4

        self.scale_heads = []
        for in_feature in self.in_features:
            head_distance = int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            in_channels = feature_channels[in_feature]
            head_ops_list = self._head_ops_list(in_channels, head_distance)
            self.scale_heads.append(nn.Sequential(*head_ops_list))
            self.add_module(in_feature, self.scale_heads[-1])

        # Not yet upsampled to the input size directly consumed by the model.
        self.predictor = nn.Conv2d(self.conv_dims, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, features, targets=None):
        """
        Returns:
        In training, returns (None, dict of losses)
        In inference, returns (Bxnum_classesxHxW logits, {})
        """
        preds = self.layers(features)
        if self.training:
            return None, self.losses(preds, targets)
        else:
            return preds, {}

    def layers(self, features):
        """in_features: {'p2':xx, 'p3':xx, ...}"""

        # Merge output feature maps from all scales
        x = None
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x += self.scale_heads[i](features[f])

        # Upsample to the image size
        x = F.interpolate(
            x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )

        # predict classification scores
        x = self.predictor(x)

        return x

    def losses(self, predictions, targets):
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

    def _head_ops_list(self, input_channels, head_distance):
        if head_distance == 0:
            return nn.ModuleList([self._pyconv4_block_no_upsample(input_channels, self.conv_dims)])
        elif head_distance == 1:
            return nn.ModuleList([self._pyconv4_block(input_channels, self.conv_dims)])
        elif head_distance == 2:
            return nn.ModuleList([self._pyconv3_block(input_channels, self.conv_dims),
                                  self._pyconv4_block(self.conv_dims, self.conv_dims)])
        else:
            assert head_distance == 3
            return nn.ModuleList([self._pyconv2_block(input_channels, self.conv_dims),
                                  self._pyconv3_block(self.conv_dims, self.conv_dims),
                                  self._pyconv4_block(self.conv_dims, self.conv_dims)])

    def _pyconv2_block(self, in_channels, out_channels):
        pyconv2_block = nn.Sequential(PyConv2(in_channels, out_channels),
                                      get_norm(self.norm, out_channels),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                                      )
        return pyconv2_block

    def _pyconv3_block(self, in_channels, out_channels):
        pyconv3_block = nn.Sequential(PyConv3(in_channels, out_channels),
                                      get_norm(self.norm, out_channels),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                                      )
        return pyconv3_block

    def _pyconv4_block(self, in_channels, out_channels):
        pyconv4_block = nn.Sequential(PyConv4(in_channels, out_channels),
                                      get_norm(self.norm, out_channels),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                                      )
        return pyconv4_block

    def _pyconv4_block_no_upsample(self, in_channels, out_channels):
        pyconv4_block = nn.Sequential(PyConv4(in_channels, out_channels),
                                      get_norm(self.norm, out_channels),
                                      nn.ReLU())
        return pyconv4_block
