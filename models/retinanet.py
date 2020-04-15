import torch
from torchvision import models

import nn
import math

from anchors import FpnAnchor


presets = {
    'default': {
        'box_sizes': (1, math.pow(2., 1./3.), math.pow(2., 2./3.)),
        'ratios': (2., 1., .5),
    },
    'retinanet': {
        'inherit': 'retinanet-50-500-voc'
    },
    'retinanet-50-500-voc': {
        'inherit': 'retinanet-50-500',
        'num_class': 21
    },
    'retinanet-50-500': {
        'width': 500,
        'backbone': 'resnet50',
        'pyramid_size': 63
    },
    'retinanet-50-600-voc': {
        'inherit': 'retinanet-50-600',
        'num_class': 21
    },
    'retinanet-50-600': {
        'width': 600,
        'backbone': 'resnet50',
        'pyramid_size': 75
    },
    'retinanet-101-500-voc': {
        'inherit': 'retinanet-101-500',
        'num_class': 21
    },
    'retinanet-101-500': {
        'width': 500,
        'backbone': 'resnet101',
        'pyramid_size': 63
    },
    'retinanet-101-600-voc': {
        'inherit': 'retinanet-101-600',
        'num_class': 21
    },
    'retinanet-101-600': {
        'width': 600,
        'backbone': 'resnet101',
        'pyramid_size': 75
    },
}


class CrossScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(size)

    def forward(self, x, y):
        return self.conv(x) + self.upsample(y)


class RetinaNet(nn.Module):
    def __init__(self, preset='retinanet', params=None, pretrained=False):
        super().__init__()

        self.name = preset

        self.top_down_layers = None
        self.bottom_up_layers = None

        self.classifiers = None
        self.box_regressions = None

        # merge network parameters
        self.params = {}
        self.apply_params(presets['default'])
        self.apply_params(presets[preset])
        self.apply_params(params)

        p = self.params

        self.num_class = p['num_class']

        # build anchor
        self.anchor = FpnAnchor(p['pyramid_size'],
                                5,
                                p['box_sizes'],
                                p['ratios'])

        # build top-down/bottom-up pathway
        self.build_pyramid(pretrained=pretrained)

        # build regression layers
        self.build_regressions()

        # initialize weight/bias
        self.initialize_parameters()

    def forward(self, x):
        bottom_up = []

        # bottom-up pathway
        for layer in self.bottom_up_layers:
            x = layer(x)
            bottom_up.append(x)

        # top-down pathway
        x = bottom_up.pop()

        top_down = [x]

        for layer in self.top_down_layers:
            x = layer(bottom_up.pop(), x)
            top_down.append(x)

        batch_size = int(x.shape[0])

        conf = []
        loc = []

        for i, x in enumerate(top_down):
            y = self.classifiers(x).permute(0, 2, 3, 1)
            conf.append(y.reshape(batch_size, -1, self.num_class))

            y = self.box_regressions[i](x).permute(0, 2, 3, 1)
            loc.append(y.reshape(batch_size, -1, 4))

        conf = torch.cat(conf, dim=1)
        loc = torch.cat(loc, dim=1)

        return conf, loc

    def get_anchor_box(self):
        return self.anchor

    def get_input_size(self):
        return self.params['width'], self.params['width']

    def build_pyramid(self, pretrained):
        # build bottom-up pathway
        self.build_bottom_up(pretrained)
        
        # build top-down pathway
        self.build_top_down()

    def build_bottom_up(self, pretrained):
        backbone = self.params['backbone']

        if backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            model = models.resnet101(pretrained=pretrained)
        else:
            raise Exception("unimplemented backbone %s" % backbone)

        # p3 ~ p5 are extracted from backbone
        p3 = nn.Sequential(model.conv1, 
                           model.bn1,
                           model.relu,
                           model.maxpool,
                           model.layer1,
                           model.layer2)

        p4 = model.layer3
        p5 = model.layer4

        # build remaining layers
        in_channels = self.calc_in_channel_width(p5)
        p6 = nn.Conv2d(in_channels, 256, 3, stride=2, padding=1)

        p7 = nn.Sequential(nn.ReLU(),
                           nn.Conv2d(256, 256, 3, stride=2, padding=1))

        # register bottom up layers
        self.bottom_up_layers = nn.ModuleList((p3, p4, p5, p6, p7))

    def build_top_down(self):
        top_down_layers = []

        # ignore size of p1, p2
        size = self.params['width']
        for i in range(2):
            size = int((size + 1) / 2) 

        # size of p3, p4, p5, p6 ,p7
        sizes = []
        for i in range(len(self.bottom_up_layers)):
            sizes.append(size)
            size = int((size + 1) / 2) 

        for i in range(len(self.bottom_up_layers), 1, -1):
            layer = self.bottom_up_layers[i-2]

            in_channels = self.calc_in_channel_width(layer)
            top_down_layers.append(CrossScaleBlock(in_channels, 256, sizes[i-1]))

        self.top_down_layers = nn.ModuleList(top_down_layers)

    def build_regressions(self):
        box_regressions = []

        num_box = len(self.params['box_sizes']) * len(self.params['ratios'])
        num_class = self.num_class

        out_channels = num_box * num_class
        classifiers = nn.Sequential(nn.Conv2dReLU(256, 256, 3, 1, 1),
                                    nn.Conv2dReLU(256, 256, 3, 1, 1),
                                    nn.Conv2dReLU(256, 256, 3, 1, 1),
                                    nn.Conv2dReLU(256, 256, 3, 1, 1),
                                    nn.Conv2d(256, out_channels, 3, 1, 1),
                                    nn.Sigmoid())

        out_channels = num_box * 4
        for i in range(len(self.top_down_layers)+1):
            regression = nn.Sequential(nn.Conv2dReLU(256, 256, 3, 1, 1),
                                       nn.Conv2dReLU(256, 256, 3, 1, 1),
                                       nn.Conv2dReLU(256, 256, 3, 1, 1),
                                       nn.Conv2dReLU(256, 256, 3, 1, 1),
                                       nn.Conv2d(256, out_channels, 3, 1, 1))

            box_regressions.append(regression)

        self.classifiers = classifiers
        self.box_regressions = nn.ModuleList(box_regressions)

    def initialize_parameters(self):
        self.initialize_weight(self.bottom_up_layers[2:])
        self.initialize_weight(self.top_down_layers)

        self.initialize_weight(self.classifiers)
        self.initialize_weight(self.box_regressions)

    def initialize_weight(self, layer):
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def calc_in_channel_width(self, prev):
        if type(prev).__name__ == 'Sequential':
            return self.calc_in_channel_width(prev[-1])
        elif type(prev).__name__ == 'Bottleneck':
            return prev.bn3.num_features
        elif type(prev).__name__ == 'Conv2d':
            return prev.out_channels
        else:
            raise Exception("failed to guess input channel width")

    def apply_params(self, params):
        if params is None:
            return

        if 'inherit' in params.keys():
            self.apply_params(presets[params['inherit']])

        for k, v in params.items():
            if k == 'inherit':
                continue

            self.params[k] = v


def build_retinanet(preset, params=None, pretrained=False):
    return RetinaNet(preset, params, pretrained)


