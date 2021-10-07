import torch

from torch import nn
from torchvision.ops import nms


class DetectPostProcess(nn.Module):
    def __init__(self, anchor, th_conf=0.5, th_iou=0.5, classifier='softmax'):
        super().__init__()

        self.anchor = anchor

        self.th_conf = th_conf
        self.th_iou = th_iou

        self.classifier = classifier
        self.softmax = None if classifier == 'softmax' else nn.Softmax(dim=2)

    def forward(self, conf, loc):
        num_cls = conf.size(2)

        if self.classifier == 'softmax':
            score = self.softmax(conf)
        else:
            score = conf.sigmoid()

        return self.nms(self.anchor, num_cls, score, loc)

    def nms(self, anchor, num_cls, score, loc):
        th_conf = self.th_conf
        th_iou = self.th_iou

        batch_size = score.size(0)

        box = anchor.decode(loc)

        batches = []
        for b in range(0, batch_size):
            cls_base = 0 if self.classifier == 'sigmoid' else 1
            classes = []

            for i in range(cls_base, num_cls):
                mask = score[b][:, i] >= th_conf

                _box = box[b][mask]
                _score = score[b][mask, i]

                idx = nms(_box, _score, th_iou)
                objs = torch.cat((_box[idx], _score[idx].unsqueeze(1)), 1)

                classes.append(objs.tolist())

            batches.append(classes)

        return batches

