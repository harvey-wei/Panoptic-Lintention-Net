import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from typing import Dict
from detectron2.config import CfgNode as CN

from projects.Panoptic_Attention.panoptic_attention import UpsampleAtt, Pixel2PatchAttention
from detectron2.layers import (
    ShapeSpec,
    get_norm,
)

class ToySemSegHead(nn.Module):
    def __init__(self,
                 cfg=None,
                 in_channels=256,
                 num_classes=80,
                 input_shape: Dict[str, ShapeSpec]=None,
                 ):
        super().__init__()
        if cfg is not None:
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

        self.loss_weight = 0.5
        self.scale_heads = []
        # print(f'type of in_channels {type(in_channels)}')
        # channel reduce -> [ppa->x2]x2
        # (N, C, H, W) -> (N, C//2, H, W) -> (N, 2*H, 2*W, C//2) -> (N, , 4*H, 4*W, C//2)-> (N, 8*H, 8*W, C//2)
        level_p5 = nn.ModuleDict({
            "channel_reduce": nn.Sequential(nn.Conv2d(in_channels,
                                                      in_channels // 2,
                                                      kernel_size=1),
                                            nn.GroupNorm(32, in_channels // 2),
                                            nn.ReLU()
                                            ),
            "pp_attention": nn.Sequential(Pixel2PatchAttention(in_channels // 2,
                                                               num_patches=16,
                                                               query_project=True,
                                                               patches_project=True,
                                                               position_embed=True,
                                                               ),
                                          UpsampleAtt(),
                                          Pixel2PatchAttention(in_channels // 2,
                                                               num_patches=16,
                                                               query_project=True,
                                                               patches_project=True,
                                                               position_embed=True,
                                                               ),
                                          UpsampleAtt(),
                                          Pixel2PatchAttention(in_channels // 2,
                                                               num_patches=16,
                                                               query_project=True,
                                                               patches_project=True,
                                                               position_embed=True,
                                                               ),
                                          UpsampleAtt(),
                                          )
        })

        # channel reduce -> [ppa->x2]x2
        # (N, C, 2*H, 2*W) -> (N, 2*H, 2*W, C//2) -> (N, 4*H, 4*W, C//2)-> (N, 8*H, 8*W, C//2)
        level_p4 = nn.ModuleDict({
            "channel_reduce": nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                                            nn.GroupNorm(32, in_channels // 2),
                                            nn.ReLU()),
            "pp_attention": nn.Sequential(Pixel2PatchAttention(in_channels // 2,
                                                               num_patches=16,
                                                               query_project=True,
                                                               patches_project=True,
                                                               position_embed=True,
                                                               ),
                                          UpsampleAtt(),
                                          Pixel2PatchAttention(in_channels // 2,
                                                               num_patches=16,
                                                               query_project=True,
                                                               patches_project=True,
                                                               position_embed=True,
                                                               ),
                                          UpsampleAtt()
                                          )
        })

        level_p3 = nn.ModuleDict({
            "channel_reduce": nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                                            nn.GroupNorm(32, in_channels // 2),
                                            nn.ReLU()),
            "pp_attention": nn.Sequential(Pixel2PatchAttention(in_channels // 2,
                                                               num_patches=16,
                                                               query_project=True,
                                                               patches_project=True,
                                                               position_embed=True,
                                                               ),
                                          UpsampleAtt(),
                                          )
        })

        level_p2 = nn.ModuleDict({
            "channel_reduce": nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                                            nn.GroupNorm(32, in_channels // 2),
                                            nn.ReLU()),
            "pp_attention": nn.Sequential(Pixel2PatchAttention(in_channels // 2,
                                                               num_patches=16,
                                                               query_project=True,
                                                               patches_project=True,
                                                               position_embed=True,
                                                               ),
                                          )
        })
        self.scale_heads.append(level_p2)
        self.add_module('p2', level_p2)
        self.scale_heads.append(level_p3)
        self.add_module('p3', level_p3)
        self.scale_heads.append(level_p4)
        self.add_module('p4', level_p4)
        self.scale_heads.append(level_p5)
        self.add_module('p5', level_p5)

        self.predictor = nn.Conv2d(in_channels//2, num_classes, kernel_size=1, stride=1, padding=0)

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
        feat = None
        for i, f in enumerate(features):   # i = 0,1,2,3. f = 'p2', 'p3', 'p4', 'p5'
            if i == 0:
                # feature_reduce shape (N, hh, ww, C//2)
                feature_reduce = self._modules[f]['channel_reduce'](features[f]).permute(0, 2, 3, 1)
                feat = self._modules[f]['pp_attention'](feature_reduce)

            else:
                feature_reduce = self._modules[f]['channel_reduce'](features[f]).permute(0, 2, 3, 1)
                feat += self._modules[f]['pp_attention'](feature_reduce)

        # feat -> (N, C//2, 8*H, 8*W)
        feat = feat.permute(0, 3, 1, 2).contiguous()

        # predictions shape (N, num_classes, 8*H, 8*W)
        predictions = self.predictor(feat)

        # predictions shape (N, num_classes, 32*H, 32*W) logits
        predictions = F.interpolate(predictions, scale_factor=4, mode='bilinear', align_corners=False)

        return predictions

    def losses(self, predictions, targets):
        loss = F.cross_entropy(
            predictions, targets, reduction="mean"
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

    # def layers(self, features, targets):
    #     # targets (N, 32*H, 32*W)
    #     # feat_p2 (N, C, 8*H, 8*W)
    #     # feat_p3 (N, C, 4*H, 4*W)
    #     # feat_p4 (N, C, 2*H, 2*W)
    #     # feat_p5 (N, C, H, W)
    #
    #     feat = None
    #
    #     for i, f in enumerate(features):   # i = 0,1,2,3. f = 'p2', 'p3', 'p4', 'p5'
    #         if i == 0:
    #             # feature_reduce shape (N, hh, ww, C//2)
    #             feature_reduce = self._modules[f]['channel_reduce'](features[f]).permute(0, 2, 3, 1)
    #             feat = self._modules[f]['pp_attention'](feature_reduce)
    #
    #         else:
    #             feature_reduce = self._modules[f]['channel_reduce'](features[f]).permute(0, 2, 3, 1)
    #             feat += self._modules[f]['pp_attention'](feature_reduce)
    #
    #     # feat_p4 = features['p4']
    #     #
    #     # feat_p5 = features['p5']
    #     #
    #     # # self.scale_heads[0] - level_p4
    #     # # self.scale_heads[1] - level_p5
    #     #
    #     # # feat_p4->(N, C//2, 2*H, 2*W)->(N, 2*H, 2*W, C//2)
    #     # feat_p4 = self.scale_heads[0]['channel_reduce'](feat_p4).permute(0, 2, 3, 1)
    #     #
    #     # # print(f'feat_p4 after channel reduce shape {feat_p4.size()}')
    #     # # feat_p4->(N, 8*H, 8*W, C//2)
    #     # feat = self.scale_heads[0]['pp_attention'](feat_p4)
    #     #
    #     # # feat_p5->(N, C//2, H, W)->(N, H, W, C//2)
    #     # feat_p5 = self.scale_heads[1]['channel_reduce'](feat_p5).permute(0, 2, 3, 1)
    #     #
    #     # # feat_p4->(N, 8*H, 8*W, C//2)
    #     # feat += self.scale_heads[1]['pp_attention'](feat_p5)
    #
    #     # # feat shape (N, 8*H, 8*W, C//2)
    #     # feat = feat_p4 + feat_p5
    #     #
    #     # feat -> (N, C//2, 8*H, 8*W)
    #     feat = feat.permute(0, 3, 1, 2)
    #
    #     # predictions shape (N, num_classes, 8*H, 8*W)
    #     predictions = self.predictor(feat)
    #
    #     # predictions shape (N, num_classes, 32*H, 32*W)
    #     predictions = F.interpolate(predictions, scale_factor=4, mode='bilinear', align_corners=False)
    #
    #     loss = F.cross_entropy(predictions, targets, reduction='mean')
    #
    #     return loss * 0.5


class UpsamleAttTest(unittest.TestCase):
    def setUp(self) -> None:
        # Mock the configs for the head
        cfg = CN()
        cfg.MODEL = CN()
        cfg.MODEL.SEM_SEG_HEAD = CN()
        cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegHeadSingleHeadPPA'
        # p2: x4; p3: x8; p4: x16; p5:x32
        cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
        # Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
        # the correposnding pixel.
        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
        # Number of classes in the semantic segmentation head
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54
        # Normalization method for the convolution layers. Options: "" (no norm), "GN".
        cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
        cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
        cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128

        # hyper-parameters for PPA
        cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_PATCHES = 16
        cfg.MODEL.SEM_SEG_HEAD.PPA_QUERY_PROJECT = True
        cfg.MODEL.SEM_SEG_HEAD.PPA_PATCHES_PROJECT = True
        cfg.MODEL.SEM_SEG_HEAD.PPA_POSITION_EMBED = True

        # Mock the feature maps from all levels of FPN
        # Mock the labels.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Current Device {self.device}')
        self.batch_size = 2
        self.img_height = 512
        self.img_width = 512
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.targets = torch.randint(0, self.num_classes, (self.batch_size, self.img_height, self.img_width),
                                     device=self.device
                                     )
        self.fpn_features = {}
        self.fpn_features_shape = {}
        self.fpn_features_names = ['p2', 'p3', 'p4', 'p5']

        self.fpn_channels = 256
        curr_stride = 4
        for fpn_features_name in self.fpn_features_names:
            feature_height = self.img_height // curr_stride
            feature_width = self.img_width // curr_stride
            self.fpn_features[fpn_features_name] = torch.randn(self.batch_size,
                                                               self.fpn_channels,
                                                               feature_height,
                                                               feature_width,
                                                               device=self.device
                                                               )
            self.fpn_features_shape[fpn_features_name] = ShapeSpec(channels=self.fpn_channels,
                                                                   height=feature_height,
                                                                   width=feature_width, stride=curr_stride)
            curr_stride *= 2

        # sem_seg_head_name = cfg.MODEL.SEM_SEG_HEAD.NAME
        # self.sem_seg_head = ToySemSegHead(cfg, in_channels=256, input_shape=self.fpn_features_shape)
        self.sem_seg_head = ToySemSegHead()
        self.sem_seg_head.to(device=self.device)

    def test_upsample_att(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"device: {device}")
        #
        N, C, H, W = 4, 256, 16, 16
        features = {"p5": torch.randn(N, C, H, W, device=device),
                    "p4": torch.randn(N, C, 2*H, 2*W, device=device),
                    "p3": torch.randn(N, C, 4*H, 4*W, device=device),
                    "p2": torch.randn(N, C, 8*H, 8*W, device=device),
                    }

        targets = torch.randint(0, 80,
                                (N, 32 * H, 32 * W),
                                device=device
                                )
        # sem_head = ToySemSegHead(in_channels=C, num_classes=80).to(device=device)
        # sem_head.train()
        #
        # pred, loss = sem_head(features, targets)
        # loss['loss_sem_seg'].backward()
        # print(loss)
        pred, loss = self.sem_seg_head(self.fpn_features, self.targets)
        # pred, loss = self.sem_seg_head(features, targets)
        loss['loss_sem_seg'].backward()


if __name__ == "__main__":
    unittest.main()
