import torch
import unittest
from projects.Panoptic_PyConv.panoptic_pyconv import (
    SE,
    GroupConvSE,
    VerConvSeparated,
    GConvGPool,
    PyGConvGPool,
    VerConv,
    PyConv,
)


class VersatileConvTest(unittest.TestCase):
    def setUp(self) -> None:
        b, c, h, w = 4, 32, 12, 24
        self.in_fm = torch.rand((b, c, h, w))

    def test_SE(self):
        in_channels = self.in_fm.shape[1]
        se_layer = SE(in_channels, reduction_rate=16)
        res = se_layer(self.in_fm)

        self.assertEqual(res.shape, self.in_fm.shape)

    def test_GroupConvSE(self):
        b, in_channels, h, w = self.in_fm.shape
        out_channesl = 2 * in_channels
        k = 3
        num_group = 4
        s = 1
        group_conv = GroupConvSE(in_channels,
                                 out_channesl,
                                 kernel_size=k,
                                 num_group=num_group,
                                 stride=s)
        res = group_conv(self.in_fm)
        expected_shape = (b, out_channesl, h // s, w // s)

        self.assertEqual(res.shape, expected_shape)

    def test_VerConvSeparated(self):
        b, c, h, w = self.in_fm.shape
        out_channels = 128

        strides = [1, 2, 4]
        num_levels = [1, 2, 3, 4]
        reduction_rates = [4, 8, 16]
        for reduction_rate in reduction_rates:
            for num_level in num_levels:
                for stride in strides:
                    ver_conv = VerConvSeparated(in_channels=c,
                                                out_channels=out_channels,
                                                num_levels=num_level,
                                                stride=stride,
                                                reduction_rate=reduction_rate
                                                )
                    res = ver_conv(self.in_fm)
                    self.assertEqual(res.shape, (b, out_channels, h // stride, w // stride))

    def test_gconv_gpool(self):
        b, c, h, w = self.in_fm.shape
        out_channels = 128
        k = 5
        s = 2
        g = 4

        gconv_gpool = GConvGPool(in_channels=c,
                                 out_channels=out_channels,
                                 kernel_size=k,
                                 stride=s,
                                 num_groups=g)
        res = gconv_gpool(self.in_fm)
        self.assertEqual(res['out_feature'].shape, (b, out_channels, h // s, w // s))
        self.assertEqual(res['global_feature'].shape, (b, out_channels, 1, 1))

    def test_py_gconv_gpool(self):
        b, c, h, w = self.in_fm.shape
        out_channels = 256

        strides = [1, 2, 4]
        num_levels = [1, 2, 3, 4]

        for stride in strides:
            for num_level in num_levels:
                conv = PyGConvGPool(in_channels=c,
                                    out_channels=out_channels,
                                    stride=stride,
                                    num_levels=num_level)

                res = conv(self.in_fm)
                self.assertEqual(res['out_feature'].shape, (b, out_channels, h // stride, w // stride))
                self.assertEqual(res['global_feature'].shape, (b, out_channels, 1, 1))

    def test_VerConv(self):
        b, c, h, w = self.in_fm.shape
        out_channels = 256

        strides = [1, 2, 4]
        num_levels = [1, 2, 3, 4]
        reduction_rates = [4, 8, 16]

        for stride in strides:
            for num_level in num_levels:
                for reduction_rate in reduction_rates:
                    conv = VerConv(in_channels=c,
                                   out_channels=out_channels,
                                   stride=stride,
                                   num_levels=num_level,
                                   reduction_rate=reduction_rate)
                    res = conv(self.in_fm)
                    self.assertEqual(res.shape, (b, out_channels, h // stride, w // stride))

    def test_pyconv(self):
        b, c, h, w = self.in_fm.shape
        out_channels = 256

        strides = [1, 2, 4]
        num_levels = [1, 2, 3, 4]

        for stride in strides:
            for num_level in num_levels:
                conv = PyConv(in_channels=c,
                              out_channels=out_channels,
                              stride=stride,
                              num_levels=num_level)

                res = conv(self.in_fm)
                self.assertEqual(res.shape, (b, out_channels, h // stride, w // stride))

    def test_pyconvse(self):
        b, c, h, w = self.in_fm.shape
        out_channels = [64, 128, 256, 512]

        strides = [1, 2, 4]
        num_levels = [1, 2, 3, 4]

        for stride in strides:
            for num_level in num_levels:
                for out_channel in out_channels:
                    conv = PyConv(in_channels=c,
                                  out_channels=out_channel,
                                  stride=stride,
                                  num_levels=num_level)

                    res = conv(self.in_fm)
                    self.assertEqual(res.shape, (b, out_channel, h // stride, w // stride))


if __name__ == '__main__':
    unittest.main()