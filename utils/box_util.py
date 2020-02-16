import torch
from PIL import ImageDraw


colors = (( 79, 195, 247),
          (236, 64, 122),
          (126, 87, 194),
          (205, 220, 57),
          (103, 58, 183),
          (255, 160, 0))


def calc_iou(a, b):
    dims = (a.size(0), b.size(0), 4)

    a = a.unsqueeze(1).expand(*dims)
    b = b.unsqueeze(0).expand(*dims)

    x1 = torch.max(a[..., 0], b[..., 0])
    x2 = torch.min(a[..., 2], b[..., 2])
    y1 = torch.max(a[..., 1], b[..., 1])
    y2 = torch.min(a[..., 3], b[..., 3])
  
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    intersect = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
    union = area_a + area_b - intersect

    return intersect / union


def draw_object_box(img, objects):
    draw = ImageDraw.Draw(img)

    fg = (0, 0, 0)
    bg = (128, 128, 128)

    for i, obj in enumerate(objects):
        if not obj:
            continue

        bg = colors[i % len(colors)]
        x1, y1, x2, y2, score, label = obj

        x1 = int(x1 * img.size[0])
        x2 = int(x2 * img.size[0])
        y1 = int(y1 * img.size[1])
        y2 = int(y2 * img.size[1])

        text = '%s (%.2f)' % (label, score)
        textsize = draw.textsize(text)

        x0, y0 = (x1, y1-textsize[1])
        draw.rectangle([x0, y0, x1+textsize[0], y1], fill=bg)
        draw.rectangle([x1, y1, x2, y2], outline=bg)
        draw.text([x0, y0], text, fill=fg)

    return img

