import torch
import random

from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> detector_transforms.Compose([
        >>>     detector_transforms.CenterCrop(10),
        >>>     detector_transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), torch.tensor(target)


class Convert:
    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, img, target):
        return img.convert(mode=self.mode), target


class LetterBox:
    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, img, target):
        width, height = img.size

        if width != height:
            width_ = height_ = max(width, height)

            pad_w = width_ - width
            pad_h = height_ - height

            pad = (pad_w//2, pad_h//2, pad_w-(pad_w//2), pad_h-(pad_h//2))
        
            for t in target:
                t[0] = (t[0] * width + pad[0]) / width_
                t[1] = (t[1] * height + pad[1]) / height_
                t[2] = (t[2] * width + pad[0]) / width_
                t[3] = (t[3] * height + pad[1]) / height_

            img = ImageOps.expand(img, pad, fill=self.fill)

        return img, target


class Resize:
    def __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
        width, height = img.size

        return img.resize(self.size, self.interpolation), target


class HorizontalFlip:
    def __call__(self, img, target):
        img = ImageOps.mirror(img)

        width, height = img.size

        for t in target:
            x1 = t[0]
            x2 = t[2]

            t[0] = 1.0 - x2
            t[2] = 1.0 - x1

        return img, target


class CropStub:
    def crop(self, img, offset, size):
        x1 = offset[0]
        x2 = offset[0] + size[0]
        y1 = offset[1]
        y2 = offset[1] + size[1]

        return img.crop((x1, y1, x2, y2))

    def update_target(self, target, offset, org_size, new_size):
        x_offset, y_offset = offset

        width, height = org_size
        width_, height_ = new_size

        discarded = []
        for t in target:
            x1 = max(t[0] * width - x_offset, 0)
            x2 = min(t[2] * width - x_offset, width_)

            if (x2 - x1) <= 4:
                discarded.append(t)
                continue

            y1 = max(t[1] * height - y_offset, 0)
            y2 = min(t[3] * height - y_offset, height_)

            if (y2 - y1) <= 4:
                discarded.append(t)
                continue

            t[0] = x1 / width_
            t[1] = y1 / height_
            t[2] = x2 / width_
            t[3] = y2 / height_

        for t in discarded:
            target.remove(t)

        return target


class CenterCrop(CropStub):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        org_size = img.size
        new_size = self.size

        x_offset = (org_size[0] - new_size[0]) / 2
        y_offset = (org_size[1] - new_size[1]) / 2

        offset = (x_offset, y_offset)
        
        img_ = self.crop(img, offset, new_size)

        return img_, self.update_target(target, offset, org_size, new_size)
        

class RandomApply:
    def __init__(self, t, p=0.5):
        self.t = t
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.t(img, target)

        return img, target


class RandomConvert:
    candidates = ('BGR', 'Grey')

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            mode = random.choice(self.candidates)

            if mode == 'BGR':
                b, g, r = img.split()
                img = Image.merge("RGB", (r, g, b))
            elif mode == 'Grey':
                img = img.convert(mode='L').convert(mode='RGB')
            else:
                img = img.convert(mode=mode)

        return img, target


class RandomHorizontalFlip(HorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return super().__call__(img, target)

        return img, target


class RandomLetterBox(LetterBox):
    def __init__(self, p=0.5, fill=(255, 255, 255)):
        super().__init__(fill)

        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return super().__call__(img, target)

        return img, target


class RandomScale:
    def __init__(self, p=0.5, ratio=(1.0, 2.0), interpolation=Image.LANCZOS):
        self.interpolation = interpolation

        self.p = p
        self.ratio = ratio

    def __call__(self, img, target):
        if random.random() < self.p:
            return img, target

        ratio = random.uniform(self.ratio[0], self.ratio[1])

        if random.random() < 0.5:
            width = img.size[0]
            height = int(img.size[1] * ratio)
        else:
            width = int(img.size[0] * ratio)
            height = img.size[1]

        return img.resize((width, height), self.interpolation), target


class RandomCrop(CropStub):
    def __call__(self, img, target):
        width, height = img.size

        x_min = y_min = 1.0
        x_max = y_max = 0.0

        for t in target:
            x_min = min(x_min, t[0])
            y_min = min(y_min, t[1])
            x_max = max(x_max, t[2])
            y_max = max(y_max, t[3])

        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)

        l = random.randint(0, x_min)
        r = random.randint(0, width - x_max)
        t = random.randint(0, y_min)
        b = random.randint(0, height - y_max)

        offset = (l, t)
        new_size = (width - l - r, height - t - b)

        img_ = self.crop(img, offset, new_size)
        
        return img_, self.update_target(target, offset, img.size, new_size)


class RandomSamplePatch(RandomCrop):
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        width, height = img.size

        t = random.choice(target)
        ratio = random.choice((0.1, 0.3, 0.5, 0.7, 0.9))

        x1 = int(t[0] * width)
        y1 = int(t[1] * height)
        x2 = int(t[2] * width)
        y2 = int(t[3] * height)

        w = x2 - x1
        h = y2 - y1

        x_pad = int(w / ratio)
        y_pad = int(h / ratio)

        l_pad = random.randint(0, x_pad)
        r_pad = x_pad - l_pad
        t_pad = random.randint(0, y_pad)
        b_pad = y_pad - t_pad

        _x1 = max(x1 - l_pad, 0)
        _y1 = max(y1 - t_pad, 0)
        _x2 = min(x2 + r_pad, width)
        _y2 = min(y2 + b_pad, height)

        offset = (_x1, _y1)
        new_size = (_x2 - _x1, _y2 - _y1)

        img_ = self.crop(img, offset, new_size)

        return img_, self.update_target(target, offset, img.size, new_size)


class Normalize:
    def __init__(self, mean=.5, std=.5):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        return (img - self.mean) / self.std, target

