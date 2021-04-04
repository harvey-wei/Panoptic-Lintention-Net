import torch
import unittest
import numpy as np
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.layers import (
    ShapeSpec,
)
from projects.Panoptic_PyConv.panoptic_pyconv import (
    ConvBlock,
    VerConvSeparated,
    VerConv,
    PyConvSE,
    HeadLevel,
    SemSegVerConvHead,
)


class VerConvSemSegHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        # Mock the configs for the head
        cfg = CN()
        cfg.MODEL = CN()
        cfg.MODEL.SEM_SEG_HEAD = CN()
        cfg.MODEL.SEM_SEG_HEAD.NAME = 'SemSegVerConvHead'
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
        cfg.MODEL.SEM_SEG_HEAD.REDUCT_RATE = 16  # defaults to 16
        # cfg.MODEL.SEM_SEG_HEAD.VERCONV = "VerConvSeparated"
        # cfg.MODEL.SEM_SEG_HEAD.VERCONV = "VerConv"
        cfg.MODEL.SEM_SEG_HEAD.VERCONV = "PyConvSE"

        # Mock the feature maps from all levels of FPN
        # Mock the labels.
        self.batch_size = 8
        self.img_height = 64
        self.img_width = 128
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.targets = torch.randint(0, self.num_classes, (self.batch_size, self.img_height, self.img_width))
        self.fpn_features = {}
        self.fpn_features_shape = {}
        self.fpn_features_names = ['p2', 'p3', 'p4', 'p5']

        self.fpn_channels = 256
        curr_stride = 4
        for fpn_features_name in self.fpn_features_names:
            feature_height = self.img_height // curr_stride
            feature_width = self.img_width // curr_stride
            self.fpn_features[fpn_features_name] = torch.randn(self.batch_size, self.fpn_channels,
                                                               feature_height, feature_width)
            self.fpn_features_shape[fpn_features_name] = ShapeSpec(channels=self.fpn_channels,
                                                                   height=feature_height,
                                                                   width=feature_width, stride=curr_stride)
            curr_stride *= 2

        sem_seg_head_name = cfg.MODEL.SEM_SEG_HEAD.NAME
        self.sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(sem_seg_head_name)(cfg, self.fpn_features_shape)

    def test_ConvBlock(self):
        b, c, h, w = 4, 32, 12, 24
        self.in_fm = torch.rand((b, c, h, w))

        out_channels = [64, 128, 256, 512]

        strides = [1, 2, 4]
        num_levels = [1, 2, 3, 4]

        conv_type = [VerConvSeparated, VerConv, PyConvSE]
        umsample = False

        for verconv in conv_type:
            for stride in strides:
                for num_level in num_levels:
                    for out_channel in out_channels:
                        conv = ConvBlock(conv=verconv,
                                         norm="GN",
                                         in_channels=c,
                                         out_channels=out_channel,
                                         stride=stride,
                                         num_levels=num_level,
                                         reduction_rate=8,
                                         upsample=umsample)

                        res = conv(self.in_fm)
                        if umsample:
                            self.assertEqual(res.shape, (b, out_channel, h // stride * 2, w // stride * 2))
                        else:
                            self.assertEqual(res.shape, (b, out_channel, h // stride, w // stride))

    def test_head_level(self):
        b, c, h, w = 4, 256, 64, 64,
        strides = [4, 8, 16, 32]
        comm_channels = 128
        conv_type = [VerConvSeparated, VerConv, PyConvSE]
        for conv_class in conv_type:
            for stride in strides:
                x = torch.rand(b, c, h // stride, w // stride)
                head_dist = np.log2(stride) - np.log2(4)
                head_level = HeadLevel(in_channels=c,
                                       comm_out_channels=comm_channels,
                                       conv=conv_class,
                                       norm="GN",
                                       reduction_rate=16,
                                       head_dist=head_dist
                                       )
                res = head_level(x)
                self.assertEqual(res.shape, (b, comm_channels, h // 4, w // 4))

    def test_sem_seg_head_eval(self):
        self.sem_seg_head.eval()
        preds, loss = self.sem_seg_head(self.fpn_features)
        expected_shape = (self.batch_size, self.num_classes, self.img_height, self.img_width)
        self.assertEqual(preds.shape, expected_shape, 'Output shape is incorrect!')

    def test_sem_seg_head_train(self):
        self.sem_seg_head.train()
        preds, loss = self.sem_seg_head(self.fpn_features, self.targets)
        self.assertIsInstance(loss, dict)


if __name__ == '__main__':
    unittest.main()
