import math
import torch


class DefaultBox:
    def __init__(self, num_grids, ratios, s_min=0.2, s_max=0.9):
        super().__init__()

        self.num_grids = num_grids
        self.ratios = ratios

        self.s_min = s_min
        self.s_max = s_max

        boxes = []
        num_ratios = []

        for i, _ in enumerate(num_grids):
            box, num_ratio = self.build_boxes(i)

            boxes.extend(box)
            num_ratios.append(num_ratio)

        boxes = torch.tensor(boxes).detach()

        if torch.cuda.is_available():
            boxes = boxes.cuda()

        self.default_boxes = boxes
        self.num_ratios = num_ratios

    def build_boxes(self, i):
        ratios = self.ratios[i]
        num_grid = self.num_grids[i]

        b = []
        for y in range(0, num_grid):
            for x in range(0, num_grid):
                # generate 1:1 size box
                b.append(self.build_box(i, x, y, 1.))

                # generate ratio...
                for r in ratios:
                    b.append(self.build_box(i, x, y, r))
                    b.append(self.build_box(i, x, y, 1./r))

                # generate extra size box 
                b.append(self.build_box(i, x, y))

        return b, (2 + len(ratios)*2)

    def build_box(self, i, x, y, r=None):
        ratios = self.ratios

        s_min = self.s_min
        s_max = self.s_max

        num_grid = self.num_grids[i]
        grid_width = 1. / num_grid

        s = s_min + (s_max - s_min) * i / (len(ratios)-1)
        s_ = s_min + (s_max - s_min) * (i+1) / (len(ratios)-1)

        if r is None:
            w = h = math.sqrt(s * s_)
        else:
            w = s * math.sqrt(r)
            h = s * math.sqrt(1./r)

        return [(x+0.5)*grid_width, (y+0.5)*grid_width, w, h]

    # (x1, y1, x2, y2) -> (delta_x, delta_y, delta_w, delta_h)
    def encode(self, raw):
        has_batch = len(raw.shape) == 3

        if not has_batch:
            raw = raw.unsqueeze(0)

        anchor = self.default_boxes.unsqueeze(0).expand_as(raw)

        # (x1, y1, x2, y2) -> (cx, cy, w, h)
        cxcy = (raw[:, :, 2:4] + raw[:, :, 0:2]) / 2.
        wh = raw[:, :, 2:4] - raw[:, :, 0:2]

        anchor_cxcy = anchor[:, :, 0:2]
        anchor_wh = anchor[:, :, 2:4]

        # delta_x = (x - anchor_x) / anchor_width
        delta_xy = (cxcy - anchor_cxcy) / anchor_wh

        # delta_w = ln(width / anchor_width)
        delta_wh = torch.log(wh / anchor_wh)

        encoded = torch.cat((delta_xy, delta_wh), 2)
        if not has_batch:
            encoded = encoded.squeeze(0)

        return encoded

    # (delta_x, delta_y, delta_w, delta_h) -> (x1, y1, x2, y2)
    def decode(self, encoded):
        has_batch = len(encoded.shape) == 3

        if not has_batch:
            encoded = encoded.unsqueeze(0)

        anchor = self.default_boxes.expand_as(encoded)

        # delta_x * anchor_width + anchor_x
        xy = encoded[:, :, 0:2] * anchor[:, :, 2:4] + anchor[:, :, 0:2]

        # exp(delta_w) * anchor_width
        half_wh = torch.exp(encoded[:, :, 2:4]) * anchor[:, :, 2:4] / 2.

        raw = torch.cat((xy - half_wh, xy + half_wh), 2)
        if not has_batch:
            raw = raw.squeeze(0)

        return raw

    def get_anchor(self):
        return self.default_boxes

    def get_num_ratios(self, depth):
        return self.num_ratios[depth]

