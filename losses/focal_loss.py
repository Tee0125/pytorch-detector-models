import torch

from torch.functional import F
from utils.box_util import calc_iou


class FocalLoss:
    def __init__(self, anchor, alpha=0.25, gamma=2.0):
        self.anchor = anchor

        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, truth, logit, loc):
        _coord = []
        _label = []

        _loc = []
        _logit = []

        for idx in range(0, len(truth)):
            _truth = truth[idx]

            if len(_truth) < 1:
                continue

            if torch.cuda.is_available():
                _truth = _truth.cuda()

            coord, label = self.truth2anchor(_truth)

            _coord.append(coord)
            _label.append(label)

            _loc.append(loc[idx])
            _logit.append(logit[idx])

        coord = torch.stack(_coord, 0)
        label = torch.stack(_label, 0)

        loc = torch.stack(_loc, 0)
        logit = torch.stack(_logit, 0)

        return self.calc_loss(coord, label, loc, logit)

    def truth2anchor(self, truth):
        # step1. matching strategy
        anchor = self.anchor.get_anchor()

        truth_coord = truth[:, 0:4]
        truth_label = truth[:, 4].long() + 1

        # - (cx,cy, w, h) to (x1, x2, y1, y2)
        anchor_coord = torch.cat([anchor[:, 0:2] - anchor[:, 2:4]/2.,
                                  anchor[:, 0:2] + anchor[:, 2:4]/2.], 1)

        iou = calc_iou(truth_coord, anchor_coord)

        # - max(...) per ground truth
        best_match, best_match_idx = iou.max(dim=1)

        # - max(...) per anchor
        max_iou, max_iou_idx = iou.max(dim=0)

        # - labels, coord per box
        coord = truth_coord[max_iou_idx]
        label = truth_label[max_iou_idx]

        # - set label to unknown for IOU < 0.5
        label[max_iou < 0.5] = -1

        # - set label to background for IOU < 0.4
        label[max_iou < 0.4] = 0

        # x1y1x2y2 to cxcywh
        coord = self.anchor.encode(coord)

        return coord.detach(), label.detach()

    def calc_loss(self, coord, label, loc, logit):
        # calculate focal loss
        l_logit = self.focal_loss(label, logit)
        l_loc = self.smooth_l1_loss(coord, label, loc)

        return l_logit + l_loc

    @staticmethod
    def smooth_l1_loss(coord, label, loc):
        pos_mask = label > 0

        pos_coord = coord[pos_mask]
        pos_loc = loc[pos_mask]

        loss = F.smooth_l1_loss(pos_loc, pos_coord, reduction='sum')

        return loss / pos_mask.sum()

    def focal_loss(self, label, logit):
        alpha = self.alpha
        gamma = self.gamma

        num_class = logit.size(2)

        one_hot = torch.eye(num_class)

        if torch.cuda.is_available():
            one_hot = one_hot.cuda()

        # ignore ambiguious samples
        mask = label >= 0

        label = label[mask]
        logit = logit[mask]

        target = one_hot[label][..., 1:]
        logit = logit[..., 1:]

        # calculate focal loss
        p = logit.sigmoid()
        neg_target = 1. - target

        p_t = target * p + neg_target * (1. - p)
        alpha = target * alpha + neg_target * (1. - alpha)

        modulator = torch.pow(1. - p_t, gamma)

        # logit.sigmoid() is applied in binary_cross_entropy_with_logits
        loss = F.binary_cross_entropy_with_logits(logit, target,
                                                  reduction='none')

        return torch.sum(alpha * modulator * loss) / mask.sum()
