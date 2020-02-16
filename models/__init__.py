import os
import torch

from .ssd import SSD, build_ssd


def build_model(args, params=None, pretrained=False):
    model = args.model.lower()

    if model.startswith('ssd'):
        return build_ssd(model, params, pretrained=pretrained)
    
    raise Exception("unknown model %s" % args.model)


def load_model(model, source):
    if not os.path.isfile(source):
        raise Exception("can not open checkpoint %s" % source)

    model.load_state_dict(torch.load(source))


def save_model(model, path='./checkpoints', postfix=None):
    if postfix:
        postfix = '_' + postfix
    else:
        postfix = ''

    target = os.path.join(path, model.name + postfix + '.pth')

    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save(model.state_dict(), target)

