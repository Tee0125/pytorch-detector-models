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
                 bias=True,
                 use_batchnorm=False):

        super().__init__()

        expand = nn.Conv2dReLU(in_channels,
                               in_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias,
                               use_batchnorm=use_batchnorm)

        shrink = nn.Conv2dReLU(in_channels,
                               out_channels,
                               1,
                               groups=in_channels,
                               bias=bias,
                               use_batchnorm=use_batchnorm)

        self.net = nn.Sequential(expand, shrink)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)
