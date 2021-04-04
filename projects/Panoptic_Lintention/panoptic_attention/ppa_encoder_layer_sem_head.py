import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from typing import Dict, Callable
import fvcore.nn.weight_init as weight_init

from detectron2.layers import (
    ShapeSpec,
    get_norm,
)

from projects.Panoptic_Attention.attentions import (
    PPAttention,
    MultiHeadPPAttention,
    PPALayer,
)


class UpsampleAtt(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 mode="bilinear",
                 align_corners=False
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """x is of shape (N, H, W, C)"""

        # x shape -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # x shape (N, C, H, W)
        x = F.interpolate(x,
                          scale_factor=self.scale_factor,
                          mode=self.mode,
                          align_corners=self.align_corners
                          )

        # x shape -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)

        return x


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegHeadPPALayer(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES  # [p2, p3, p4, p5]
        feature_strides = {k: v.stride for k, v in input_shape.items()}  # pi: stride
        feature_channels = {k: v.channels for k, v in input_shape.items()}  # pi: channel
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride = 4
        num_patches = cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_PATCHES
        query_project = cfg.MODEL.SEM_SEG_HEAD.PPA_QUERY_PROJECT
        patches_project = cfg.MODEL.SEM_SEG_HEAD.PPA_PATCHES_PROJECT
        position_embed = cfg.MODEL.SEM_SEG_HEAD.PPA_POSITION_EMBED
        num_heads = cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_HEADS

        assert len(self.in_features) == 4

        self.scale_heads = []
        for in_feature in self.in_features:
            head_distance = int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            in_channels = feature_channels[in_feature]
            head_ops = nn.ModuleDict()  # subclass of nn.Module to holds sub-modules in a dictionary.

            # expect input of shape (N, C, H, W)
            head_ops["channel_reduce"] = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                 out_channels=self.conv_dims,
                                                                 kernel_size=1),
                                                       nn.GroupNorm(num_groups=32, num_channels=self.conv_dims),
                                                       nn.ReLU(),
                                                       )

            if head_distance == 0:
                # expect input of shape (N, H, W, C)
                # be sure to permute the dim before feeding feature map to attention module.
                head_ops["pp_attention"] = PPALayer(d_model=self.conv_dims,
                                                    num_heads=num_heads,
                                                    num_patches=num_patches,
                                                    query_project=query_project,
                                                    patches_project=patches_project,
                                                    position_embed=position_embed,
                                                    )
            else:
                ppa_list = []
                for i in range(head_distance):
                    ppa_list.append(PPALayer(d_model=self.conv_dims,
                                             num_heads=num_heads,
                                             num_patches=num_patches,
                                             query_project=query_project,
                                             patches_project=patches_project,
                                             position_embed=position_embed,
                                             ))
                    ppa_list.append(UpsampleAtt(scale_factor=2, mode="bilinear", align_corners=False))
                # expect input of shape (N, H, W, C)
                # be sure to permute the dim before feeding feature map to attention module.
                head_ops["pp_attention"] = nn.Sequential(*ppa_list)

            self.scale_heads.append(head_ops)
            self.add_module(in_feature, self.scale_heads[-1])

        # Not yet upsampled to the input size directly consumed by the model.
        self.predictor = nn.Conv2d(self.conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        """
        Args:
            features: {"p2": tensor of shape (N, C, H, W),...}
            targets:  Detectron2-style ground truth labels.
        Returns:
            At training phase, returns tuple (None, dict of losses)
            At testing stage, returns  (tensor (N, num_classes, H_img, W_img), {})
        """
        # predictions shape (N, num_classes, H_img, W_img)
        predictions = self.layers(features)
        if self.training:
            return None, self.losses(predictions, targets)
        else:
            return predictions, {}

    def layers(self, features: Dict[str, torch.Tensor]):
        x = None
        for i, f in enumerate(self.in_features):
            if i == 0:
                # feats_reduce shape (N, C, H, W)
                feats_reduce = self.scale_heads[i]["channel_reduce"](features[f])

                # feats_reduce shape (N, H, W, C)
                # x shape (N, H, W, C)
                feats_reduce = feats_reduce.permute(0, 2, 3, 1)
                x = self.scale_heads[i]["pp_attention"](feats_reduce)
            else:
                # feats_reduce shape (N, C, H, W)
                feats_reduce = self.scale_heads[i]["channel_reduce"](features[f])

                # feats_reduce shape (N, H, W, C)
                # x shape (N, H, W, C)
                feats_reduce = feats_reduce.permute(0, 2, 3, 1)
                x += self.scale_heads[i]["pp_attention"](feats_reduce)

        # x shape -> (N, C, H_img/4, W_img/4)
        x = x.permute(0, 3, 1, 2).contiguous()

        # predictions shape (N, num_classes, H_img/4, W_img/4)
        predictions = self.predictor(x)

        # predictions shape -> (N, num_classes, H_img, W_img)
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )

        return predictions

    def losses(self, predictions, targets):
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
