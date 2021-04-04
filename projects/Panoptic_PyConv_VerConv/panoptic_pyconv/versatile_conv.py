from torch import nn
import torch


class SE(nn.Module):
    """
    Squeeze and Excitation Module according to the `paper`: Squeeze-and-Excitation Networks
    """

    def __init__(self, in_channels, reduction_rate=16):
        super().__init__()

        self.in_channels = in_channels
        self.reduction_rate = reduction_rate

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # output is of shape (b, c, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_rate, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_rate, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        out = self.avg_pool(x).view(b, c)  # (b, c)
        out = self.fc(out)  # (b, c)

        x = x * out.view(b, c, 1, 1)
        return x


class GroupConvSE(nn.Module):
    """
    A grouped convolution immediately followed by the SE module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_group,
                 stride=1,
                 reduction_rate=16
                 ):
        super().__init__()

        # Grouped convolution with half(same) padding.
        self.group_conv = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    padding=kernel_size // 2,
                                    stride=stride,
                                    groups=num_group)

        self.se_layer = SE(in_channels=out_channels, reduction_rate=reduction_rate)

    def forward(self, x):
        x = self.group_conv(x)
        x = self.se_layer(x)
        return x


class VerConvSeparated(nn.Module):
    """
    A new convolution, that we term Versatile Convolution, VerConv for short,
    with the capability of capturing feature at different scales,
    contextual information and modeling the channel interdependencies.

    For this version, grouped conv at each scale is fed into an independent SE Module.
    We term it VerConvSeparated.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 stride,
                 reduction_rate,
                 ):
        super().__init__()

        # per-level design parameters
        kernel_sizes = [3, 5, 7, 9]
        num_groups = [1, 4, 8, 16]

        assert out_channels % 4 == 0 and num_levels <= 4
        if num_levels == 4:
            pyramid_out_channels = [out_channels // 4] * 4
        elif num_levels == 3:
            pyramid_out_channels = [out_channels // 4] * 2 + [out_channels // 2]
        elif num_levels == 2:
            pyramid_out_channels = [out_channels // 2] * 2
        else:
            # num_levels = 1
            pyramid_out_channels = [out_channels]

        self.pyramid_convs = []
        for level in range(num_levels):
            self.pyramid_convs.append(GroupConvSE(
                in_channels=in_channels,
                out_channels=pyramid_out_channels[level],
                kernel_size=kernel_sizes[level],
                num_group=num_groups[level],
                stride=stride,
                reduction_rate=reduction_rate
            ))
            self.add_module("group" + str(level), self.pyramid_convs[-1])

    def forward(self, x):
        outs = []
        for conv in self.pyramid_convs:
            outs.append(conv(x))

        y = torch.cat(outs, dim=1)

        return y


class GConvGPool(nn.Module):
    """A grouped convolution followed by a global pooling"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 num_groups,
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              padding=kernel_size // 2,  # half(same) padding
                              stride=stride,
                              groups=num_groups)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        output = {}

        y = self.conv(x)
        output['out_feature'] = y  # b, c, h_in // stride, w_in // stride

        output['global_feature'] = self.pool(y)  # b, c, 1, 1

        return output


class PyGConvGPool(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 num_levels,
                 ):
        super().__init__()

        # per-level design parameters
        kernel_sizes = [3, 5, 7, 9]
        num_groups = [1, 4, 8, 16]

        assert out_channels % 4 == 0 and num_levels <= 4
        if num_levels == 4:
            pyramid_out_channels = [out_channels // 4] * 4
        elif num_levels == 3:
            pyramid_out_channels = [out_channels // 4] * 2 + [out_channels // 2]
        elif num_levels == 2:
            pyramid_out_channels = [out_channels // 2] * 2
        else:
            # num_levels = 1
            pyramid_out_channels = [out_channels]

        self.pyramid = []
        for level in range(num_levels):
            self.pyramid.append(GConvGPool(in_channels=in_channels,
                                           out_channels=pyramid_out_channels[level],
                                           kernel_size=kernel_sizes[level],
                                           stride=stride,
                                           num_groups=num_groups[level]))
            self.add_module("level" + str(level), self.pyramid[-1])

    def forward(self, x):
        out_features = []
        global_features = []

        for gconv_gpool in self.pyramid:
            y = gconv_gpool(x)
            out_features.append(y['out_feature'])
            global_features.append(y['global_feature'])

        output = {'out_feature': torch.cat(out_features, dim=1),  # (b, out_channels, h // stride, w // stride)
                  'global_feature': torch.cat(global_features, dim=1),  # (b, out_channels, 1, 1)
                  }

        return output


class VerConv(nn.Module):
    """
    Global information is collected per scale level into multiple per-level vectors.
    Then, we concatenate these vectors into one long vector, which is fed into a 2-layer fully connected net to
    learn the per-channel scaling factor.
    Finally, the concatenated output feature maps are scaled by the learned scaling factors.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 stride=1,
                 reduction_rate=16
                 ):
        super(VerConv, self).__init__()
        self.py_gconv_gpool = PyGConvGPool(in_channels=in_channels,
                                           out_channels=out_channels,
                                           stride=stride,
                                           num_levels=num_levels)

        self.fc = nn.Sequential(nn.Linear(out_channels, out_channels // reduction_rate, bias=False),
                                nn.ReLU(),
                                nn.Linear(out_channels // reduction_rate, out_channels, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.py_gconv_gpool(x)
        out_feature = x['out_feature']  # (b, out_channels, h, w)
        global_feature = x['global_feature']  # (b, out_channels, 1, 1)
        b, c, _, _ = out_feature.shape
        scales = self.fc(global_feature.view(b, c))
        out_feature = out_feature * scales.view(b, c, 1, 1)

        return out_feature


class PyConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels, stride=1):
        super().__init__()
        # per-level design parameters
        kernel_sizes = [3, 5, 7, 9]
        num_groups = [1, 4, 8, 16]

        assert out_channels % 4 == 0 and num_levels <= 4
        if num_levels == 4:
            pyramid_out_channels = [out_channels // 4] * 4
        elif num_levels == 3:
            pyramid_out_channels = [out_channels // 4] * 2 + [out_channels // 2]
        elif num_levels == 2:
            pyramid_out_channels = [out_channels // 2] * 2
        else:
            # num_levels = 1
            pyramid_out_channels = [out_channels]

        self.pyramid = []
        for level in range(num_levels):
            assert pyramid_out_channels[level] % num_groups[level] == 0
            self.pyramid.append(nn.Conv2d(in_channels,
                                          pyramid_out_channels[level],
                                          kernel_sizes[level],
                                          stride=stride,
                                          padding=kernel_sizes[level] // 2,  # half(same) padding
                                          groups=num_groups[level]))
            self.add_module('py_level' + str(level), self.pyramid[-1])

    def forward(self, x):
        outs = []

        for pyconv in self.pyramid:
            outs.append(pyconv(x))

        return torch.cat(outs, dim=1)


class PyConvSE(nn.Module):
    """PyConv is immediately followed by a SE module. """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 stride=1,
                 reduction_rate=16):
        super().__init__()

        self.pyconv = PyConv(in_channels=in_channels,
                             out_channels=out_channels,
                             num_levels=num_levels,
                             stride=stride
                             )

        self.SE_layer = SE(in_channels=out_channels, reduction_rate=reduction_rate)

    def forward(self, x):
        x = self.pyconv(x)  # (b, out_channels, h, w)
        x = self.SE_layer(x)  # (b, out_channels, h, w)

        return x
