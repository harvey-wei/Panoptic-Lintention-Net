import torch
from torch import nn
import torch.nn.functional as F
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from typing import Dict
import numpy as np
import fvcore.nn.weight_init as weight_init
from typing import Callable
from .versatile_conv import (
    VerConvSeparated,
    VerConv,
    PyConvSE,
)

from detectron2.layers import (
    ShapeSpec,
    get_norm,
)


class ConvBlock(nn.Module):
    """conv->norm->relu->[upsample]
        conv should be any of VerConvSeparated,VerConv, PyConvSE
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv: Callable,
                 norm: str,
                 num_levels,
                 stride=1,
                 reduction_rate=2,
                 upsample=True
                 ):
        super().__init__()

        block = [conv(in_channels=in_channels,
                      out_channels=out_channels,
                      num_levels=num_levels,
                      stride=stride,
                      reduction_rate=reduction_rate
                      ),
                 get_norm(norm, out_channels),
                 nn.ReLU()]

        if upsample:
            block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)


class HeadLevel(nn.Module):
    """Each block maintains the spatial size"""

    def __init__(self,
                 in_channels,
                 comm_out_channels,
                 conv: Callable,
                 norm,
                 reduction_rate,
                 head_dist):
        super().__init__()

        chain = None
        if head_dist == 0:
            chain = [ConvBlock(in_channels=in_channels,
                               out_channels=comm_out_channels,
                               conv=conv,
                               norm=norm,
                               num_levels=4,
                               reduction_rate=reduction_rate,
                               upsample=False)]
        elif head_dist == 1:
            chain = [ConvBlock(in_channels=in_channels,
                               out_channels=comm_out_channels,
                               conv=conv,
                               norm=norm,
                               num_levels=4,
                               reduction_rate=reduction_rate,
                               upsample=True)]
        elif head_dist == 2:
            chain = [ConvBlock(in_channels=in_channels,
                               out_channels=comm_out_channels,
                               conv=conv,
                               norm=norm,
                               num_levels=3,
                               reduction_rate=reduction_rate,
                               upsample=True),
                     ConvBlock(in_channels=comm_out_channels,
                               out_channels=comm_out_channels,
                               conv=conv,
                               norm=norm,
                               num_levels=4,
                               reduction_rate=reduction_rate,
                               upsample=True)]
        elif head_dist == 3:
            chain = [ConvBlock(in_channels=in_channels,
                               out_channels=comm_out_channels,
                               conv=conv,
                               norm=norm,
                               num_levels=2,
                               reduction_rate=reduction_rate,
                               upsample=True),
                     ConvBlock(in_channels=comm_out_channels,
                               out_channels=comm_out_channels,
                               conv=conv,
                               norm=norm,
                               num_levels=3,
                               reduction_rate=reduction_rate,
                               upsample=True),
                     ConvBlock(in_channels=comm_out_channels,
                               out_channels=comm_out_channels,
                               conv=conv,
                               norm=norm,
                               num_levels=4,
                               reduction_rate=reduction_rate,
                               upsample=True)]

        self.head_level = nn.Sequential(*chain)

    def forward(self, x):
        return self.head_level(x)


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegVerConvHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES  # p2, p3, p4, ....
        feature_strides = {k: v.stride for k, v in input_shape.items()}  # pi: stride
        feature_channels = {k: v.channels for k, v in input_shape.items()}  # pi: channel
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride = 4
        reduction_rate = cfg.MODEL.SEM_SEG_HEAD.REDUCT_RATE  # defaults to 16
        ver_conv_type = cfg.MODEL.SEM_SEG_HEAD.VERCONV
        ver_conv = {"VerConvSeparated": VerConvSeparated,
                    "VerConv": VerConv,
                    "PyConvSE": PyConvSE,
                    }[ver_conv_type]

        self.scale_heads = []
        for in_feature in self.in_features:
            head_distance = int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            in_channels = feature_channels[in_feature]
            head_ops = HeadLevel(in_channels=in_channels,
                                 comm_out_channels=self.conv_dims,
                                 conv=ver_conv,
                                 norm=norm,
                                 reduction_rate=reduction_rate,
                                 head_dist=head_distance)
            self.scale_heads.append(head_ops)
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
