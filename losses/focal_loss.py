import torch

from torch.functional import F
from utils.box_util import calc_iou


class FocalLoss:
    def __init__(self, anchor, alpha=0.25, gamma=2.0):
        self.anchor = anchor

        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, truth, conf, loc):
        _coord = []
        _label = []

        _loc = []
        _conf = []

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
            _conf.append(conf[idx])

        coord = torch.stack(_coord, 0)
        label = torch.stack(_label, 0)

        loc = torch.stack(_loc, 0)
        conf = torch.stack(_conf, 0)

        return self.calc_loss(coord, label, loc, conf)

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

        # - link ground truth to best matched anchor box
        for truth_idx, anchor_idx in enumerate(best_match_idx):
            max_iou[anchor_idx] = 1.
            max_iou_idx[anchor_idx] = truth_idx

        # - labels, coord per box
        coord = truth_coord[max_iou_idx]
        label = truth_label[max_iou_idx]

        # - set label to unknown for conf < 0.5
        label[max_iou < 0.5] = -1

        # - set label to background for conf < 0.4
        label[max_iou < 0.4] = 0

        # x1y1x2y2 to cxcywh
        coord = self.anchor.encode(coord)

        return coord.detach(), label.detach()

    def calc_loss(self, coord, label, loc, conf):
        # calculate focal loss
        l_conf = self.focal_loss(label, conf)
        l_loc = self.smooth_l1_loss(coord, label, loc)

        return l_conf + l_loc

    @staticmethod
    def smooth_l1_loss(coord, label, loc):
        pos_mask = label > 0

        pos_coord = coord[pos_mask]
        pos_loc = loc[pos_mask]

        loss = F.smooth_l1_loss(pos_loc, pos_coord, reduction='sum')

        return loss / pos_mask.sum()

    def focal_loss(self, label, conf):
        alpha = self.alpha
        gamma = self.gamma

        num_class = conf.size(2)

        one_hot = torch.eye(num_class)

        if torch.cuda.is_available():
            one_hot = one_hot.cuda()

        # ignore ambiguious samples
        mask = label >= 0

        label = label[mask]
        conf = conf[mask]

        tf_map = one_hot[label][..., 1:]
        conf = conf[..., 1:]

        # calculate focal loss
        score = conf.sigmoid()

        p_t = tf_map * score + (1 - tf_map) * (1. - score)
        alpha = tf_map * alpha + (1 - tf_map) * (1. - alpha)

        weight = torch.pow(1. - p_t, gamma) * alpha

        # conf.sigmoid() is applied in binary_cross_entropy_with_logits
        loss = F.binary_cross_entropy_with_logits(conf, tf_map,
                                                  weight.detach(),
                                                  reduction='sum')

        return loss / mask.sum()
