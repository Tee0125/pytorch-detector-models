import torch
from torchvision import models

import nn
from anchors import DefaultBox


presets = {
    'default': {
        's_min': 0.15,
        's_max': 0.95,
        's_extra_min': None,
    },

    'ssdlite': {
        'inherit': 'ssdlite-mobilenetv2-voc'
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
            (2.,3.), (2., 3.), (2., 3.), (2.,3.), (2.,3.)
        ),
        'num_grids': (20, 10, 5, 3, 1),
    },
}


class SSDLite(nn.Module):
    def __init__(self, preset='ssdlite', params=None, pretrained=False):
        super().__init__()

        self.name = preset

        self.extras = None
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
                                      p['s_extra_min'])

        # setup backbone
        self.build_backbone(pretrained)

        # build extra layers
        self.build_extras()

        # build regression layers
        self.build_regressions()

        # initialize weight/bias
        self.initialize_parameters()

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

        batch_size = x.shape[0]

        conf = []
        loc = []

        for i, x in enumerate(pyramid):
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

            in_channels = out_channels

        extra = nn.Sequential(*extra)
        extra.out_channels = out_channels

        return extra, out_channels

    def build_regressions(self):
        classifiers = []
        box_regressions = []

        extras = [self.b0]
        extras.extend(self.extras)

        # from extras
        for i, extra in enumerate(extras):
            in_channels = self.calc_in_channel_width(extra)
            n = self.default_box.get_num_ratios(i)

            classifier = nn.Conv2d(in_channels, n * self.num_class, 3, 1, 1)
            classifiers.append(classifier)

            box_regression = nn.Conv2d(in_channels, n * 4, 3, 1, 1)
            box_regressions.append(box_regression)

        in_channels = self.calc_in_channel_width(self.b0)
        l2_norm = nn.Norm2d(in_channels)

        self.l2_norm = l2_norm
        self.classifiers = nn.ModuleList(classifiers)
        self.box_regressions = nn.ModuleList(box_regressions)

    def build_backbone(self, pretrained):
        backbone = self.params['backbone']

        if backbone == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
            features = model.features

            b0 = nn.Sequential(features[0:14], features[14].conv[0])
            b1 = nn.Sequential(features[14].conv[1:], features[15:])

            b0.out_channels = 576
            b1.out_channels = 1280

        else:
            raise Exception("not implemented backbone %s" % backbone)

        in_channels = self.calc_in_channel_width(b0)

        self.b0 = b0
        self.b1 = b1

        self.in_channels = in_channels

    def initialize_parameters(self):
        self.init_parameters(self.extras)
        self.init_parameters(self.classifiers)
        self.init_parameters(self.box_regressions)

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
    def init_parameters(layer):
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def calc_in_channel_width(prev):
        if not hasattr(prev, 'out_channels'):
            raise Exception("failed to guess input channel width")

        return prev.out_channels


def build_ssdlite(preset='ssdlite', params=None, pretrained=False):
    return SSDLite(preset, params=params, pretrained=pretrained)

