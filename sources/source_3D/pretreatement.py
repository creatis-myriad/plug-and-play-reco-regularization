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

# ircad files
#toutes les images
median_size = 10

img_file = "../../datas/ircad_iso_V3"
images = sorted(glob(os.path.join(img_file, "3Dircadb*/maskedLiverIso.nii")))
gts = sorted(glob(os.path.join(img_file, "3Dircadb*/vesselsIso.nii")))
masks = sorted(glob(os.path.join(img_file, "3Dircadb*/liverMaskIso.nii")))
file_pretreated_results = sorted(os.listdir("../../datas/ircad_iso_V3/pretreated_ircad_"+ str(median_size)))
files = [{"source_2D": img, "label": gt, "mask" : mask} for img, gt, mask in zip(images, gts,masks)]

device = "cpu"

def substract_background(image_path, mask_path, kernel_radius):
    image = image_utils.read_nifti_image(image_path)
    mask = image_utils.read_nifti_image(mask_path)
    image = image_utils.normalize_image(image)
    mask = image_utils.normalize_image(mask)

    image = (image * 255).astype(np.uint8)

    ball = morphology.ball(kernel_radius)
    background = filters.median(image, ball)
    background = image.astype(np.int16) - background
    background[background < 0] = 0
    mask = morphology.binary_erosion(mask, morphology.ball(4))

    image_pre_processed = background * mask

    return image_pre_processed

