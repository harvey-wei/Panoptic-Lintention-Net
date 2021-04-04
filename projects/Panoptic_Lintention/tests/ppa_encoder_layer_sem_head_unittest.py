import torch
import unittest
import numpy as np
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.layers import (
    ShapeSpec,
)

from projects.Panoptic_Attention.panoptic_attention import SemSegHeadPPALayer


class SemSegHeadPPATest(unittest.TestCase):
    def setUp(self):
        # Mock the configs for the head
        cfg = CN()
        cfg.MODEL = CN()
        cfg.MODEL.SEM_SEG_HEAD = CN()
        cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegHeadPPALayer'
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
        cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_PATCHES = 16
        cfg.MODEL.SEM_SEG_HEAD.PPA_QUERY_PROJECT = True,
        cfg.MODEL.SEM_SEG_HEAD.PPA_PATCHES_PROJECT = True,
        cfg.MODEL.SEM_SEG_HEAD.PPA_POSITION_EMBED = True,
        cfg.MODEL.SEM_SEG_HEAD.PPA_NUM_HEADS = 4
        cfg.MODEL.SEM_SEG_HEAD.PPA_DROPOUT = 0.1,
        cfg.MODEL.SEM_SEG_HEAD.PPA_FFN_EXPANSION = 4,

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mock the feature maps from all levels of FPN
        # Mock the labels.
        self.batch_size = 8
        self.img_height = 64
        self.img_width = 128
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
            self.fpn_features[fpn_features_name] = torch.randn(self.batch_size, self.fpn_channels,
                                                               feature_height, feature_width,
                                                               device=self.device
                                                               )
            self.fpn_features_shape[fpn_features_name] = ShapeSpec(channels=self.fpn_channels,
                                                                   height=feature_height,
                                                                   width=feature_width, stride=curr_stride)
            curr_stride *= 2

        sem_seg_head_name = cfg.MODEL.SEM_SEG_HEAD.NAME
        self.sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(sem_seg_head_name)(cfg, self.fpn_features_shape)
        self.sem_seg_head = self.sem_seg_head.to(device=self.device)

    def test_sem_seg_head_eval(self):
        self.sem_seg_head.eval()
        preds, loss = self.sem_seg_head(self.fpn_features)
        expected_shape = (self.batch_size, self.num_classes, self.img_height, self.img_width)
        self.assertEqual(preds.shape, expected_shape, 'Output shape is incorrect!')

    def test_sem_seg_head_train(self):
        self.sem_seg_head.train()
        # print(self.sem_seg_head)
        preds, loss = self.sem_seg_head(self.fpn_features, self.targets)
        # print(loss)
        self.assertIsInstance(loss, dict)
        loss['loss_sem_seg'].backward()


if __name__ == "__main__":
    unittest.main()