from collections import OrderedDict

import numpy as np
import torch


def random_crop(imgs, out=64):
    """
        args:
        imgs: torch.Tensor shape (B,C,H,W)
        out: output size (e.g. 64)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = torch.empty(n, c, out, out, dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11 : h11 + out, w11 : w11 + out]
    return cropped


def center_crop(image, output_size=64):
    h, w = image.shape[-2], image.shape[-1]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    if image.dim() == 3:
        image = image[:, top : top + new_h, left : left + new_w]
    elif image.dim() == 4:
        image = image[:, :, top : top + new_h, left : left + new_w]
    else:
        raise ValueError(f"Wrong number of dimensions: {image.dim()}")

    return image


def get_augmentations(augs):
    augs_to_func = {"crop": random_crop}
    augs_func = OrderedDict()
    for aug in augs:
        augs_func[aug] = augs_to_func[aug]
    return augs_func
