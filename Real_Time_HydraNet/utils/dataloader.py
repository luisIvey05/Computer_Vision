from torch.utils.data import Dataset
from PIL import Image
import os
import random
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Usual dtypes for common modalities
KEYS_TO_DTYPES = {
    "segm": torch.long,
    "mask": torch.long,
    "depth": torch.float,
    "normals": torch.float,
}


class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
        Given mean: (R, G, B) and std: (R, G, B),
        will normalise each channel of the torch.*Tensor, i.e.
        channel = (scale * channel - mean) / std

        Args:
            scale (float): Scaling constant.
            mean (sequence): Sequence of means for R,G,B channels respecitvely.
            std (sequence): Sequence of standard deviations for R,G,B channels
                respecitvely.
            depth_scale (float): Depth divisor for depth annotations.

        """

    def __init__(self, scale, mean, std, depth_scale=1.0):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def __call__(self, sample):
        sample["image"] = (self.scale * sample["image"] - self.mean) / self.std
        if "depth" in sample:
            sample["depth"] = sample["depth"] / self.depth_scale
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): Desired output size.

    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        sample["image"] = image[top : top + new_h, left : left + new_w]
        for msk_key in msk_keys:
            sample[msk_key] = sample[msk_key][top : top + new_h, left : left + new_w]
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample["image"] = torch.from_numpy(image.transpose((2, 0, 1)))
        for msk_key in msk_keys:
            sample[msk_key] = torch.from_numpy(sample[msk_key]).to(
                KEYS_TO_DTYPES[msk_key]
            )
        return sample


class RandomMirror(object):
    """Randomly flip the image and the mask"""

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        do_mirror = np.random.randint(2)
        if do_mirror:
            sample["image"] = cv2.flip(image, 1)
            for msk_key in msk_keys:
                scale_mult = [-1, 1, 1] if "normal" in msk_key else 1
                sample[msk_key] = scale_mult * cv2.flip(sample[msk_key], 1)
        return sample


class JointDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.img_path, self.dep_path, self.seg_path = data_path
        self.transform = transform
        self.masks_names = ("segm", "depth")

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        output = {}
        output["image"] = np.array(Image.open(self.img_path[idx]))
        output["segm"] = np.array(Image.open(self.dep_path[idx]))
        output["depth"] = np.array(Image.open(self.seg_path[idx]))
        if self.transform:
            output["names"] = self.masks_names
            output = self.transform(output)
            if "names" in output:
                del output["names"]
        return output

def preprocess(img_path, seg_path, dep_path):
    img_scale = 1.0 / 255
    depth_scale = 5000.0

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    normalise_params = [img_scale, img_mean.reshape((1, 1, 3)), img_std.reshape((1, 1, 3)), depth_scale, ]
    transform_common = [Normalise(*normalise_params), ToTensor()]

    crop_size = 400

    transform_train = transforms.Compose([RandomMirror(), RandomCrop(crop_size)] + transform_common)
    transform_val = transforms.Compose(transform_common)

    return transform_train, transform_val