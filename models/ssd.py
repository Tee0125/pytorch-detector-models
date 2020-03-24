import torch
from torchvision import models

import nn
from anchors import DefaultBox


presets = {
    'default': {
        's_MIN': None,
        's_min': 0.2,
        's_max': 0.9,
        's_extra': None,

        'use_batchnorm': False
    },

    'ssd300': {
        'inherit': 'ssd300-bn-voc'
    },

    'ssdlite': {
        'inherit': 'ssdlite-mobilenetv2-voc'
    },

    'ssd300-voc': {
        'width': 300,
        'num_class': 21,
        'backbone': 'vgg16',
        'extras': (
            # type, output_channels, kernel_size, (stride), (padding)
            (('c', 1024, 3, 1, 1), ('c', 1024, 1, 1)), # 19x19
            (('c',  256, 3, 1, 1), ('c',  512, 1, 2)), # 10x10
            (('c',  128, 3, 1, 1), ('c',  256, 1, 2)), # 5x5
            (('c',  128, 3, 1, 0), ('c',  256, 1, 1)), # 3x3
            (('c',  128, 3, 1, 0), ('c',  256, 1, 1))  # 1x1
        ),
        'ratios': (
            (2.,), (2., 3.), (2., 3.), (2., 3.), (2.,), (2.,)
        ),
        'num_grids': (38, 19, 10, 5, 3, 1),

        's_extra': 0.1,
    },

    'ssd300-bn-voc': {
        'inherit': 'ssd300-voc',
        'use_batchnorm': True
    },

    'ssdlite-mobilenetv2-voc': {
        'width': 320,
        'num_class': 21,
        'backbone': 'mobilenet_v2',
        'extras': (
            # type, output_channels, kernel_size, (stride), (padding)
            (('i',  256, 3, 1, 1), ('i',  512, 1, 1)), # 10x10
            (('i',  128, 3, 1, 1), ('i',  256, 1, 2)), # 5x5
            (('i',  128, 3, 1, 0), ('i',  256, 1, 1)), # 3x3
            (('i',  128, 3, 1, 0), ('i',  256, 1, 1))  # 1x1
        ),
        'ratios': (
            (2.,), (2., 3.), (2., 3.), (2.,), (2.,)
        ),
        'num_grids': (20, 10, 5, 3, 1),
        'use_batchnorm': True
    },
}


class SSD(nn.Module):
    def __init__(self, preset='ssd300', params=None, pretrained=False):
        super().__init__()

        self.name = preset

        # to avoid PyCharm warning
        self.extras = None
        self.l2_norms = None
        self.classifiers = None
        self.box_regressions = None
        self.in_channels = None

        # merge network parameters
        self.params = {}
        self.apply_params(presets['default'])
        self.apply_params(presets[preset])
        self.apply_params(params)

        p = self.params

        self.num_class = p['num_class']

        # build anchor
        self.default_box = DefaultBox(p['num_grids'],
                                      p['ratios'],
                                      p['s_min'],
                                      p['s_max'],
                                      p['s_extra'])

        # setup backbone
        self.build_backbone(pretrained)

        # build extra layers
        self.build_extras()

        # build regression layers
        self.build_regressions()

    def forward(self, x):
        pyramid = []

        # scale0
        x = self.b0(x)
        pyramid.append(x)

        # scale1, ...
        x = self.b1(x)

        for extra in self.extras:
            x = extra(x)
            pyramid.append(x)

        batch_size = int(x.shape[0])

        conf = []
        loc = []

        for i, x in enumerate(pyramid):
            x = self.l2_norms[i](x)

            y = self.classifiers[i](x).permute(0, 2, 3, 1)
            conf.append(y.reshape(batch_size, -1, self.num_class))

            y = self.box_regressions[i](x).permute(0, 2, 3, 1)
            loc.append(y.reshape(batch_size, -1, 4))

        conf = torch.cat(conf, dim=1)
        loc = torch.cat(loc, dim=1)

        return conf, loc

    def get_anchor_box(self):
        return self.default_box

    def get_input_size(self):
        return self.params['width'], self.params['width']

    def build_extras(self):
        in_channels = self.calc_in_channel_width(self.b1)

        extras = []
        for layers in self.params['extras']:
            extra, in_channels = self.build_extra(in_channels, layers)

            extras.append(extra)

        self.extras = nn.ModuleList(extras)

    def build_extra(self, in_channels, layers):
        use_bn = self.params['use_batchnorm']

        extra = []
        for layer in layers:
            out_channels = layer[1]

            if layer[0] == 'c':
                extra.append(nn.Conv2dReLU(in_channels,
                                           out_channels,
                                           *layer[2:],
                                           use_batchnorm=use_bn))

            elif layer[0] == 'i':
                extra.append(nn.InvertedBottleneck(in_channels,
                                                   out_channels,
                                                   *layer[2:],
                                                   use_batchnorm=use_bn))
            else:
                raise Exception("Extra layer config is broken")

            self.init_parameters(extra)

            in_channels = out_channels

        return nn.Sequential(*extra), out_channels

    def build_regressions(self):
        l2_norms = []

        classifiers = []
        box_regressions = []

        extras = [self.b0]
        extras.extend(self.extras)

        # from extras
        for i, extra in enumerate(extras):
            in_channels = self.calc_in_channel_width(extra)
            n = self.default_box.get_num_ratios(i)

            l2_norms.append(nn.Norm2d(in_channels))

            classifier = nn.Conv2d(in_channels, n * self.num_class, 3, 1, 1)
            classifiers.append(classifier)

            box_regression = nn.Conv2d(in_channels, n * 4, 3, 1, 1)
            box_regressions.append(box_regression)

        self.init_parameters(classifiers)
        self.init_parameters(box_regressions)

        self.l2_norms = nn.ModuleList(l2_norms)
        self.classifiers = nn.ModuleList(classifiers)
        self.box_regressions = nn.ModuleList(box_regressions)

    def build_backbone(self, pretrained):
        backbone = self.params['backbone']

        if backbone == "vgg16":
            if self.params['use_batchnorm']:
                model = models.vgg16_bn(pretrained=pretrained)

                b0 = model.features[0:33]
                b1 = model.features[33:-1]

                in_channels = model.features[30].out_channels
            else:
                model = models.vgg16(pretrained=pretrained)

                b0 = model.features[0:23]
                b1 = model.features[23:-1]

                in_channels = model.features[21].out_channels

            for layer in model.features:
                if isinstance(layer, nn.MaxPool2d):
                    layer.ceil_mode = True

        elif backbone == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
            features = model.features

            b0 = nn.Sequential(features[0:14], features[14].conv[0])
            b1 = nn.Sequential(features[14].conv[1:], features[15:])

            in_channels = features[14].conv[0].out_channels

        else:
            raise Exception("unimplemented backbone %s" % backbone)

        self.b0 = b0
        self.b1 = b1

        self.in_channels = in_channels

    def apply_params(self, params):
        if params is None:
            return

        if 'inherit' in params.keys():
            self.apply_params(presets[params['inherit']])

        for k, v in params.items():
            if k == 'inherit':
                continue

            self.params[k] = v

    @staticmethod
    def init_parameters(layers):
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Conv2dReLU):
                SSD.init_parameters(layer.net)
            elif isinstance(layer, nn.InvertedBottleneck):
                SSD.init_parameters(layer.net)

    @staticmethod
    def calc_in_channel_width(prev):
        for i in range(-1, -3, -1):
            if isinstance(prev[i], nn.Conv2d):
                return prev[i].out_channels
            elif isinstance(prev[i], nn.Conv2dReLU):
                return prev[i].out_channels
            elif isinstance(prev[i], nn.InvertedBottleneck):
                return prev[i].out_channels
            elif isinstance(prev[i], nn.BatchNorm2d):
                return prev[i].num_features
            elif isinstance(prev[i], models.mobilenet.ConvBNReLU):
                return prev[i][1].num_features

        raise Exception("failed to guess input channel width")


def build_ssd(preset='ssd300', params=None, pretrained=False):
    return SSD(preset, params=params, pretrained=pretrained)

