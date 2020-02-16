import torch

from torch.functional import F
from utils.box_util import calc_iou


class MultiBoxLoss:
    def __init__(self, anchor, alpha=1.0, th_iou=0.5):
        self.anchor = anchor

        self.alpha = alpha
        self.th_iou = th_iou

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

        # - set label to background for conf < th_iou
        label[max_iou < self.th_iou] = 0

        # x1y1x2y2 to cxcywh
        coord = self.anchor.encode(coord)

        return coord.detach(), label.detach()

    def calc_loss(self, coord, label, loc, conf):
        # step2. hard negative mining
        num_class = conf.size(2)

        coord = coord.view(-1, 4)
        loc = loc.view(-1, 4)

        label = label.view(-1)
        conf = conf.view(-1, num_class)

        pos_mask = label != 0
        neg_mask = label == 0

        pos_conf = conf[pos_mask]
        neg_conf = conf[neg_mask]

        pos_label = label[pos_mask]
        neg_label = label[neg_mask]

        num_pos = pos_conf.size(0)
        num_neg = min(num_pos*3, neg_conf.size(0))
        
        mined = F.log_softmax(neg_conf, 1)[:, 0].sort()[1][0:num_neg]

        neg_conf = neg_conf[mined]
        neg_label = neg_label[mined]

        # - calc l_conf
        conf = torch.cat([pos_conf, neg_conf], 0)
        label = torch.cat([pos_label, neg_label], 0)

        l_conf = F.cross_entropy(conf, label, reduction='sum')

        # - calc l_loc
        coord = coord[pos_mask]
        loc = loc[pos_mask]

        l_loc = F.smooth_l1_loss(loc, coord, reduction='sum')

        return (l_conf + self.alpha * l_loc) / num_pos

