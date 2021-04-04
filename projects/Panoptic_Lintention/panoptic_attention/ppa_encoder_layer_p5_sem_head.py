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
    Conv2d,
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
class SemSegHeadMultiHeadPPALayerP5(nn.Module):
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
        ppa_dropout = cfg.MODEL.SEM_SEG_HEAD.PPA_DROPOUT
        ppa_ffn_expansion = cfg.MODEL.SEM_SEG_HEAD.PPA_FFN_EXPANSION
        assert len(self.in_features) == 4

        # assert len(self.in_features) == 4
        self.scale_heads = []
        # print(f'type of in_channels {type(in_channels)}')
        # channel reduce -> [ppa->x2]x2
        # (N, C, H, W) -> (N, C//2, H, W) -> (N, 2*H, 2*W, C//2) -> (N, , 4*H, 4*W, C//2)-> (N, 8*H, 8*W, C//2)
        level_p5 = nn.ModuleDict({
            "channel_reduce": nn.Sequential(nn.Conv2d(feature_channels['p5'],
                                                      self.conv_dims,
                                                      kernel_size=1),
                                            nn.GroupNorm(32, self.conv_dims),
                                            nn.ReLU()
                                            ),
            "pp_attention": nn.Sequential(PPALayer(self.conv_dims,
                                                   num_heads=num_heads,
                                                   num_patches=num_patches,
                                                   query_project=query_project,
                                                   patches_project=patches_project,
                                                   position_embed=position_embed,
                                                   dropout=ppa_dropout,
                                                   ffn_expansion=ppa_ffn_expansion,
                                                   ),
                                          UpsampleAtt(scale_factor=2, mode='bilinear', align_corners=False),
                                          PPALayer(self.conv_dims,
                                                   num_heads=num_heads,
                                                   num_patches=num_patches,
                                                   query_project=query_project,
                                                   patches_project=patches_project,
                                                   position_embed=position_embed,
                                                   dropout=ppa_dropout,
                                                   ffn_expansion=ppa_ffn_expansion,
                                                   ),
                                          UpsampleAtt(scale_factor=2, mode='bilinear', align_corners=False),
                                          PPALayer(self.conv_dims,
                                                   num_heads=num_heads,
                                                   num_patches=num_patches,
                                                   query_project=query_project,
                                                   patches_project=patches_project,
                                                   position_embed=position_embed,
                                                   dropout=ppa_dropout,
                                                   ffn_expansion=ppa_ffn_expansion,
                                                   ),
                                          UpsampleAtt(scale_factor=2, mode='bilinear', align_corners=False),
                                          )
        })

        norm_module = nn.GroupNorm(32, self.conv_dims) if norm == "GN" else None

        # up to 1/4 scale (N, conv_dim, 1/4 , 1/4)
        level_p4 = nn.Sequential(Conv2d(in_channels=feature_channels['p4'],
                                        out_channels=self.conv_dims,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=not norm,
                                        norm=norm_module,
                                        activation=F.relu,
                                        ),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 Conv2d(in_channels=self.conv_dims,
                                        out_channels=self.conv_dims,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=not norm,
                                        norm=norm_module,
                                        activation=F.relu,
                                        ),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 )

        # up to 1/4 scale (N, conv_dim, 1/4, 1/4)
        level_p3 = nn.Sequential(Conv2d(in_channels=feature_channels['p3'],
                                        out_channels=self.conv_dims,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=not norm,
                                        norm=norm_module,
                                        activation=F.relu,
                                        ),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 )

        # level_p2 is at 1/4.  (N, conv_dim, 1/4, 1/4)
        level_p2 = nn.Sequential(Conv2d(in_channels=feature_channels['p2'],
                                        out_channels=self.conv_dims,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=not norm,
                                        norm=norm_module,
                                        activation=F.relu,
                                        )
                                 )

        self.scale_heads.append(level_p2)
        self.add_module('p2', level_p2)
        self.scale_heads.append(level_p3)
        self.add_module('p3', level_p3)
        self.scale_heads.append(level_p4)
        self.add_module('p4', level_p4)
        self.scale_heads.append(level_p5)
        self.add_module('p5', level_p5)

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
        # targets (N, 32*H, 32*W)
        # feat_p2 (N, C, 8*H, 8*W)
        # feat_p3 (N, C, 4*H, 4*W)
        # feat_p4 (N, C, 2*H, 2*W)
        # feat_p5 (N, C, H, W)
        feat = self._modules['p2'](features['p2'])
        feat += self._modules['p3'](features['p3'])
        feat += self._modules['p4'](features['p4'])

        # feature_reduce (N, 1/16, 1/16, conv_dim)
        feature_reduce = self._modules['p5']['channel_reduce'](features['p5']).permute(0, 2, 3, 1)

        # feat (N, conv_dim, 1/4, 1.4)
        feat += self._modules['p5']['pp_attention'](feature_reduce).permute(0, 3, 1, 2)

        # predictions shape (N, num_classes, 1/4, 1/4)
        predictions = self.predictor(feat)

        # predictions shape (N, num_classes, 1, 1) logits
        predictions = F.interpolate(predictions, scale_factor=4, mode='bilinear', align_corners=False)

        return predictions

    def losses(self, predictions, targets):
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
