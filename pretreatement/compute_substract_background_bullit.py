import numpy as np
from sources import image_utils, nifti_image
from monai.data import write_nifti
import skimage
from skimage import morphology
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(description='process parameters of the training')
parser.add_argument('dataset', type=str, default="bullit", help='numéro dimage')

args = parser.parse_args()
#toutes les images
median_size = 10
dataset = args.dataset
if dataset == "bullit":
    # img_file = "/home/carneiro/Documents/datas/Bullit_iso/2018-Sanchez/Bullit/"
    img_file = "/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/"
    name_file = f"pretreated_{median_size}"
    result_path = f"{img_file}{name_file}"
else:
    # img_file = "/home/carneiro/Documents/datas/Bullit_iso/2018-Sanchez/Bullit/"
    img_file = "/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/"
    name_file = f"pretreated_{median_size}_rician"
    result_path = f"{img_file}{name_file}"
if name_file not in os.listdir(img_file):
    os.mkdir(result_path)

name_files = sorted(glob(os.path.join(img_file, "Normal*")))
name_volumes = []
for i in name_files:
    name_volumes.append(i.split("/")[-1])
print(name_volumes)

for file_vol in name_volumes:
    print(file_vol)
    if dataset == "bullit":
        image_path = f"{img_file}{file_vol}/dataIso.nii.gz"
        gt_path = f"{img_file}{file_vol}/binaryVesselsIso_S.nii.gz"
        mask_path = f"{img_file}{file_vol}/brainMaskIso.nii.gz"
    else:
        image_path = f"{img_file}{file_vol}/rician_4.0.nii.gz"
        gt_path = f"{img_file}{file_vol}/binaryVesselsIso_S.nii.gz"
        mask_path = f"{img_file}{file_vol}/brainMaskIso.nii.gz"

    image = (image_utils.normalize_image(nifti_image.read_nifti(image_path))*255).astype(np.uint8)
    mask = image_utils.normalize_image(nifti_image.read_nifti(mask_path))
    gt = image_utils.normalize_image(nifti_image.read_nifti(gt_path))

    ball = morphology.ball(median_size)
    background = skimage.filters.median(image.copy(), ball)
    background = image.astype(np.int16) - background
    background[background < 0] = 0

    image_pre_processed = background.copy() * mask.copy()
    image_pre_processed = (image_utils.normalize_image(image_pre_processed) * 255).astype(np.uint8)
    file_result = f"{result_path}/{file_vol}"
    if file_vol not in os.listdir(result_path):
        os.mkdir(file_result)

    write_nifti(data = image, file_name = f"{file_result}/input.nii.gz", resample = False)
    write_nifti(data = (mask * 255).astype(np.uint8), file_name =f"{file_result}/mask.nii.gz", resample = False)
    write_nifti(data = image_pre_processed, file_name =f"{file_result}/image_pre_processed.nii.gz", resample = False)
    write_nifti(data = (gt * 255).astype(np.uint8), file_name =f"{file_result}/gt.nii.gz", resample = False)
