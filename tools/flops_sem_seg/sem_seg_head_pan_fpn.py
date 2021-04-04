from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.layers import (
    ShapeSpec,
)

from detectron2.modeling.meta_arch.semantic_seg import SemSegFPNHead
from projects.Panoptic_Attention.panoptic_attention import (
    SemSegHeadSingleHeadPPA,
    SemSegHeadMultiHeadPPALayerP5P4,
    SemSegHeadSingleHeadPPA2xLin,
)
from projects.Panoptic_PyConv.panoptic_pyconv import (
    SemSegPyConvHeadV1,
    SemSegVerConvHead,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import flop_count
import logging


class SemSegHeadNet(nn.Module):
    def __init__(self):
        super(SemSegHeadNet, self).__init__()
        # Mock the configs for the head
        cfg = CN()
        cfg.MODEL = CN()
        cfg.MODEL.SEM_SEG_HEAD = CN()
        # cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegFPNHead'
        # cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegHeadSingleHeadPPA'
        # cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegVerConvHead'
        # cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegPyConvHeadV1'
        cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegHeadSingleHeadPPA2xLin'
        # cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegHeadMultiHeadPPALayerP5P4'
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
        cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_HEADS = 4
        cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_PATCHES = 32
        cfg.MODEL.SEM_SEG_HEAD.PPA_QUERY_PROJECT = True
        cfg.MODEL.SEM_SEG_HEAD.PPA_PATCHES_PROJECT = True
        cfg.MODEL.SEM_SEG_HEAD.PPA_POSITION_EMBED = True
        cfg.MODEL.SEM_SEG_HEAD.PPA_DROPOUT = 0.1
        cfg.MODEL.SEM_SEG_HEAD.PPA_FFN_EXPANSION = 4
        cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4

        # hyper-parameters for SemSegMultiScalePyConvHead
        cfg.MODEL.SEM_SEG_HEAD.PYCONV_OUT_SIZE_LOCAL_CONTEXT = 512
        cfg.MODEL.SEM_SEG_HEAD.PYCONV_OUT_SIZE_GLOBAL_CONTEXT = 512
        cfg.MODEL.SEM_SEG_HEAD.PYCONV_MERGE_OUT_SIZE = 256
        cfg.MODEL.SEM_SEG_HEAD.PYCONV_LOCAL_REDUCTION = 4
        cfg.MODEL.SEM_SEG_HEAD.PYCONV_GLOBAL_BINS = 9
        cfg.MODEL.SEM_SEG_HEAD.PYCONV_CLS_DROPOUT = 0.1
        # Determine how to merge the outputs from PyConvSemHeads at different FPN levels.
        # "SUM" means summing up them while "CAT" stands for concatenating them along the depth dimension.
        cfg.MODEL.SEM_SEG_HEAD.PYCONV_FUSE_MODE = "SUM"
        # The variant of the Versatile Convolution. Any of conv_type = [VerConvSeparated, VerConv, PyConvSE]
        cfg.MODEL.SEM_SEG_HEAD.VERCONV = "VerConv"
        # The reduction rate in the fc layer of the SE-like module.
        cfg.MODEL.SEM_SEG_HEAD.REDUCT_RATE = 16

        # Mock the feature maps from all levels of FPN
        # Mock the labels.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f'Current Device {self.device}')
        self.batch_size = 1
        self.img_height = 960
        self.img_width = 1280
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

        sem_seg_head_name = cfg.MODEL.SEM_SEG_HEAD.NAME
        print(f"Sem_Seg_Head Name: {sem_seg_head_name}")
        self.sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(sem_seg_head_name)(cfg, self.fpn_features_shape)
        # self.sem_seg_head = SemSegHeadSingleHeadPPA(cfg, self.fpn_features_shape)
        self.sem_seg_head = self.sem_seg_head.to(device=self.device)

    def forward(self, dummy_input):
        self.sem_seg_head.eval()

        preds, loss = self.sem_seg_head(self.fpn_features)

        return preds, loss

def flops_sem_seg_head():
    sem_seg_net =SemSegHeadNet()
    x = torch.rand(1, 256, 800, 800)
    ret = flop_count(sem_seg_net, (x,))
    # compatible with change in fvcore
    if isinstance(ret, tuple):
        ret = ret[0]
    print(ret)
    # print(sum(ret.values()))
    print(f'Total (G)Flops: {sum(ret.values())}')


if __name__ == "__main__":
    # ############################## sem_seg_net test ##############################
    # sem_seg_net = SemSegHeadNet()
    # x = torch.rand(1, 256, 800, 800)
    # preds, loss = sem_seg_net(x)
    # print(preds.size())
    # fpn_gen = FPNGenerator()
    # fpn_features, fpn_shape = fpn_gen(x)
    # fpn_features_names = ['p2', 'p3', 'p4', 'p5']
    #
    # for fpn_feature_name in fpn_features_names:
    #     print(f'{fpn_feature_name} shape {fpn_features[fpn_feature_name].size()}')
    #     print(f'{fpn_feature_name} recorded shape  {fpn_shape[fpn_feature_name]}')
    flops_sem_seg_head()
