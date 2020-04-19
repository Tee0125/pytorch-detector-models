from .multi_box_loss import MultiBoxLoss
from .focal_loss import FocalLoss


def build_loss(args, anchor=None):
    if args.force_use_multiboxloss:
        return MultiBoxLoss(anchor, args.th_iou)
    elif args.force_use_focalloss:
        return FocalLoss(anchor)

    model = args.model.lower()

    if model.startswith('ssd'):
        return MultiBoxLoss(anchor, args.th_iou)
    elif model.startswith('retinanet'):
        return FocalLoss(anchor)

    raise Exception("unknown model %s" % args.model)
