from torch import nn


class Conv2dReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 use_batchnorm=False):

        super().__init__()

        if use_batchnorm:
            bias = False

        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias)

        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            net = nn.Sequential(conv2d, nn.BatchNorm2d(out_channels), relu)
        else:
            net = nn.Sequential(conv2d, relu)

        self.net = net

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)


class InvertedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 expand_ratio=1.,
                 bias=True,
                 use_batchnorm=True):

        super().__init__()

        layers = []

        if not bias or use_batchnorm:
            bias = False

        ex_channels = int(expand_ratio * out_channels)

        # expand
        layers.append(nn.Conv2d(in_channels, ex_channels, 1, bias=bias))

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(ex_channels))

        layers.append(nn.ReLU6(inplace=True))

        # repeat
        layers.append(nn.Conv2d(ex_channels,
                                ex_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=ex_channels,
                                bias=bias))

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(ex_channels))

        layers.append(nn.ReLU6(inplace=True))

        # shrink
        layers.append(nn.Conv2d(ex_channels, out_channels, 1, bias=bias))

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU6(inplace=True))

        self.net = nn.Sequential(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 use_batchnorm=True):

        super().__init__()

        layers = []

        # dw
        layers.append(nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=in_channels,
                                bias=bias))

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))

        layers.append(nn.ReLU6(inplace=True))

        # 1x1
        layers.append(nn.Conv2d(in_channels, out_channels, 1, bias=bias))

        self.net = nn.Sequential(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)
