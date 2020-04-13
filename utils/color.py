from PIL import Image
import numpy as np


def img2arr(img):
    return np.array(img, dtype=np.float)


def arr2img(arr):
    return Image.fromarray(np.uint8(arr))


# original code from scikit-image
def rgb2hsv(rgb):
    hsv = np.empty_like(rgb)

    # to avoid divide by zero
    old_settings = np.seterr(invalid='ignore')

    # V channel
    v = rgb.max(-1)

    # S channel
    delta = rgb.ptp(-1)

    s = delta / v
    s[delta == 0.] = 0.

    # H cannel
    # - red is max
    idx = (rgb[:, :, 0] == v)
    hsv[idx, 0] = (rgb[idx, 1] - rgb[idx,2]) / delta[idx]

    # - green is max
    idx = (rgb[:, :, 1] == v)
    hsv[idx, 0] = 2. + (rgb[idx, 2] - rgb[idx, 0]) / delta[idx]

    # - blue is max
    idx = (rgb[:, :, 2] == v)
    hsv[idx, 0] = 4. + (rgb[idx, 0] - rgb[idx, 1]) / delta[idx]

    h = (hsv[:, :, 0] / 6.) % 1.
    h[delta == 0.] = 0.

    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v

    hsv[np.isnan(hsv)] = 0.

    np.seterr(**old_settings)

    return hsv


def hsv2rgb(hsv):
    hi = np.floor(hsv[:, :, 0] * 6)

    f = hsv[:, :, 0] * 6 - hi
    p = hsv[:, :, 2] * (1 - hsv[:, :, 1])
    q = hsv[:, :, 2] * (1 - f * hsv[:, :, 1])
    t = hsv[:, :, 2] * (1 - (1 - f) * hsv[:, :, 1])
    v = hsv[:, :, 2]

    hi = np.dstack([hi, hi, hi]).astype(np.uint8) % 6
    rgb = np.choose(hi, [np.dstack((v, t, p)),
                         np.dstack((q, v, p)),
                         np.dstack((p, v, t)),
                         np.dstack((p, q, v)),
                         np.dstack((t, p, v)),
                         np.dstack((v, p, q))])

    return rgb
