import numpy as np
from monai.data import Dataset, DataLoader, write_nifti
from skimage import morphology, filters
from glob import glob
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ToTensord,
    ScaleIntensityd,
    MaskIntensityd,
    AddChanneld,
)
import sys
sys.path.insert(0, "../")
import image_utils


def substract_background(image_path, mask_path, kernel_radius):
    image = image_utils.read_nifti_image(image_path)
    mask = image_utils.read_nifti_image(mask_path)
    image = image_utils.normalize_image(image, 1)
    mask = image_utils.normalize_image(mask, 1)

    image = (image * 255).astype(np.uint8)

    ball = morphology.ball(kernel_radius)
    print("hello 1")
    background = filters.median(image, ball)
    print("hello 2")
    background = image.astype(np.int16) - background
    background[background < 0] = 0
    mask = morphology.binary_erosion(mask, morphology.ball(4))

    image_pre_processed = background * mask

    return image_pre_processed

