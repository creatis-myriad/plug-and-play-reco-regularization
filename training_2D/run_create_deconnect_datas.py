import numpy as np
import os
from glob import glob
import imageUtils
import torch
from PIL import Image
from monai.data import Dataset,DataLoader
from monai.data.utils import partition_dataset

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord
)
from personnal_transforms import BinaryDeconnect, AddArtefacts
import argparse
# parser = argparse.ArgumentParser(description='process parameters of the training')
# parser.add_argument('taille_deco_max', type=int, default="1", help='numero de limage traite entre 1 et 40')
# parser.add_argument('bruit', type=int, default="reconnect", help='path to the model')
#
#
# args = parser.parse_args()
# taille_deco_max = args.taille_deco_max
# bruit = args.bruit
# os.system(f"cp -r dataset dataset_{taille_deco_max}_{bruit}")
#
# ################################# Create binary deconnections #########################################################
# img_file = f"dataset_{taille_deco_max}_{bruit}/deconnexions"
# img_init = f"dataset_{taille_deco_max}_{bruit}/origine"
#
# mask = f"dataset_{taille_deco_max}_{bruit}/mask_fov"
# images = sorted(glob(os.path.join(img_init, "seg*.png")))
# masks = sorted(glob(os.path.join(mask, "mask*.png")))
#
#
#
# train_files = [{"image": img, "label_with_art": img,"label": img, "mask": msk} for img, msk in zip(images, masks)]
# print(train_files)
# train_trans = Compose(
#     [
#         LoadImaged(keys=["image", "label_with_art","label", "mask"]),
#         ScaleIntensityd(keys=["image", "label_with_art", "label", "mask"]),
#         BinaryDeconnect(keys=["image"], nb_deco=100, taille_max=taille_deco_max),
#         AddArtefacts(keys=["image", "label_with_art"], label="label", mask="mask",  mean_taches=bruit, std_taches=25), #100
#         # ToTensord(keys=["image", "label_with_art", "label"])
#     ]
#     )
# check_ds= Dataset(data = train_files, transform=train_trans)
# check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
#
# i = 0
# for j in range(4):
#     for batch_data in check_loader:
#         inputs, labels_with_art, labels = (
#             batch_data["image"],
#             batch_data["label_with_art"], batch_data["label"]
#         )
#         print(f"-------------------- Création de l'image n° {i} --------------------")
#
#         inputs = imageUtils.normalizeImage(inputs.squeeze().numpy(), 255)
#         inputs = np.rot90(inputs, j, [0, 1])
#         Image.fromarray(inputs.astype("uint8")).save(os.path.join(img_file, f"img_{i:d}.png"))
#
#         labels_with_art = imageUtils.normalizeImage(labels_with_art.squeeze().numpy(), 255)
#         labels_with_art = np.rot90(labels_with_art, j, [0, 1])
#         Image.fromarray(labels_with_art.astype("uint8")).save(os.path.join(img_file, f"seg_{i:d}.png"))
#
#         fragments = labels_with_art - inputs
#         Image.fromarray(fragments.astype("uint8")).save(os.path.join(img_file, f"deco_{i:d}.png"))
#
#
#         labels = imageUtils.normalizeImage(labels.squeeze().numpy(), 255)
#         labels = np.rot90(labels, j, [0, 1])
#         Image.fromarray(labels.astype("uint8")).save(os.path.join(img_file, f"label_{i:d}.png"))
#         i = i + 1

################################################### create masks ####################################################

from skimage.morphology import binary_dilation, disk
boule = disk(2)
fragments_files = glob("dataset_12/deconnexions/deco_*.png")
print(fragments_files)
for gt_path in fragments_files:
    print(f"************************** {gt_path} *****************************")
    gt = imageUtils.readImage(gt_path)
    gt = binary_dilation(gt, boule)
    num = gt_path.split("/")[-1]

    labels_with_art = imageUtils.normalizeImage(gt*1.0, 255)
    Image.fromarray(labels_with_art.astype("uint8")).save(f"dataset_12/pos_{num}")

# #### create deconnected images with noise ########
# img_file = "dataset_12/deconnexions"
# # img_init = "dataset/origine"
# for i in range(80):
#     deco = imageUtils.readImage(f"dataset_12/deconnexions/deco_{i}.png")
#     label = imageUtils.readImage(f"dataset_12/deconnexions/label_{i}.png")
#     denoise_deconnected = ((label - deco) > 0.5 ) * 1.0 *255
#     Image.fromarray(denoise_deconnected.astype("uint8")).save(os.path.join(img_file, f"denoise_deconnected_{i:d}.png"))
