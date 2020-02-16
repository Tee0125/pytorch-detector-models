import torch

from torch import nn
from torchvision.ops import nms


class DetectPostProcess(nn.Module):
    def __init__(self, anchor):
        super().__init__()

        self.anchor = anchor
        self.softmax = nn.Softmax(dim=2)

    def forward(self, conf, loc, th_iou=0.5, th_conf=0.5):
        num_cls = conf.size(2)

        score = self.softmax(conf)

        return self.nms(self.anchor, num_cls, score, loc, th_conf, th_iou)

    @staticmethod
    def nms(anchor, num_cls, score, loc, th_conf, th_iou):
        batch_size = score.size(0)

        box = anchor.decode(loc)

        batches = []
        for b in range(0, batch_size):
            classes = []

            for i in range(1, num_cls):
                mask = score[b][:, i] >= th_conf

                _box = box[b][mask]
                _score = score[b][mask, i]

                idx = nms(_box, _score, th_iou)

                _l = _box[idx].tolist()
                _s = _score[idx].tolist()

                objs = []
                for j in range(len(_l)):
                    obj = [_l[j][k] for k in range(4)]
                    obj.append(_s[j])

                    objs.append(obj)

                classes.append(objs)

            batches.append(classes)

        return batches

