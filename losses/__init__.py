from .multi_box_loss import MultiBoxLoss


def build_loss(args, anchor=None):
    model = args.model.lower()

    if model.startswith('ssd'):
        return MultiBoxLoss(anchor, args.th_iou)

    raise Exception("unknown model %s" % args.model)
