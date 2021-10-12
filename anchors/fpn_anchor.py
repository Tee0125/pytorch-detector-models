import torch
import math


class FpnAnchor:
    def __init__(self, pyramid_size, depth, box_sizes, ratios):
        grid_sizes = []
        for i in range(depth):
            grid_sizes.insert(0, pyramid_size)
            pyramid_size = (pyramid_size + 1) // 2

        scale = grid_sizes[0]

        boxes = []
        for grid_size in grid_sizes:
            step = 1.0 / grid_size
            size = scale * step

            for j in range(grid_size):
                cy = (j + 0.5) * step

                for i in range(grid_size):
                    cx = (i + 0.5) * step

                    for b in box_sizes:
                        for r in ratios:
                            r = math.sqrt(r)

                            w = size * b * r
                            h = size * b / r

                            boxes.append([cx, cy, w, h])

        boxes = torch.tensor(boxes).detach()

        if torch.cuda.is_available():
            boxes = boxes.cuda()

        self.anchors = boxes

    # (x1, y1, x2, y2) -> (delta_x, delta_y, delta_w, delta_h)
    def encode(self, raw):
        has_batch = len(raw.shape) == 3

        if not has_batch:
            raw = raw.unsqueeze(0)

        anchor = self.anchors.unsqueeze(0).expand_as(raw)

        # (x1, y1, x2, y2) -> (cx, cy, w, h)
        cxcy = (raw[:, :, 2:4] + raw[:, :, 0:2]) / 2.
        wh = raw[:, :, 2:4] - raw[:, :, 0:2]

        anchor_cxcy = anchor[:, :, 0:2]
        anchor_wh = anchor[:, :, 2:4]

        # delta_x = (x - anchor_x) / anchor_width
        delta_xy = (cxcy - anchor_cxcy) / anchor_wh / 1.

        # delta_w = ln(width / anchor_width)
        delta_wh = torch.log(wh / anchor_wh) / 1.

        encoded = torch.cat((delta_xy, delta_wh), 2)
        if not has_batch:
            encoded = encoded.squeeze(0)

        return encoded

    # (delta_x, delta_y, delta_w, delta_h) -> (x1, y1, x2, y2)
    def decode(self, encoded):
        has_batch = len(encoded.shape) == 3

        if not has_batch:
            encoded = encoded.unsqueeze(0)

        anchor = self.anchors.expand_as(encoded)

        # delta_x * anchor_width + anchor_x
        xy = encoded[:, :, 0:2] * 1. * anchor[:, :, 2:4] + anchor[:, :, 0:2]

        # exp(delta_w) * anchor_width
        half_wh = torch.exp(encoded[:, :, 2:4] * 1.) * anchor[:, :, 2:4] / 2.

        raw = torch.cat((xy - half_wh, xy + half_wh), 2)
        if not has_batch:
            raw = raw.squeeze(0)

        return raw

    def get_anchor(self):
        return self.anchors

