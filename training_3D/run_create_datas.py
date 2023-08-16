import numpy as np
import logging
import os
import sys
from glob import glob
import torch
import monai
from monai.data import Dataset, DataLoader, write_nifti
from monai.inferers import sliding_window_inference
from personnal_transforms import AddArtefacts, BinaryDeconnect, BinaryDeconnect_2, Binaries
import matplotlib.pyplot as plt

from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
    ThresholdIntensityd,
)
import argparse

# parser = argparse.ArgumentParser(description='process parameters of the training')
# parser.add_argument('dataset', type=str, default=1, help='dataset selectionné')
#
# args = parser.parse_args()
#
# if args.dataset =="Brava":
#     img_file = "/home/carneiro/Documents/datas/BraVa/GT"
#     gts = sorted(glob(os.path.join(img_file, "MRA*")))
#     saved_file = "brava_deco"
#     print(len(gts))
#     for i in range(len(gts)):
#         print(gts[i])
#     file = [{"label": gt, "deco": gt, "label_art": gt} for gt in gts]
#     print(file)
#     train_trans = Compose(
#         [
#             LoadImaged(keys=["deco", "label", "label_art"]),
#             ScaleIntensityd(keys=["deco", "label", "label_art"]),
#             BinaryDeconnect(keys=["deco"], nb_deco=120, taille_max_deco=8),
#             AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=30, seuil=0.8),
#             AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=35, seuil=0.8),
#             AddChanneld(keys=["deco", "label", "label_art"]),
#             ToTensord(keys=["deco", "label", "label_art"]),
#         ]
#     )
# elif args.dataset == "Brava_2":
#     img_file = "/home/carneiro/Documents/datas/BraVa/GT"
#     gts = sorted(glob(os.path.join(img_file, "MRA*")))
#     saved_file = "test_brava_deco_2"
#     print(len(gts))
#     for i in range(len(gts)):
#         print(gts[i])
#     file = [{"label": gt, "deco": gt, "label_art": gt} for gt in gts]
#     print(file)
#     train_trans = Compose(
#         [
#             LoadImaged(keys=["deco", "label", "label_art"]),
#             ScaleIntensityd(keys=["deco", "label", "label_art"]),
#             BinaryDeconnect_2(keys=["deco"], nb_deco=120, taille_max_deco=8),
#             AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=35, seuil=0.85),
#             AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=30, seuil=0.9),
#             # AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=35, seuil=0.8),
#             AddChanneld(keys=["deco", "label", "label_art"]),
#             ToTensord(keys=["deco", "label", "label_art"]),
#         ]
#     )
# elif args.dataset == "vascusynth":
#     img_file = "/home/carneiro/Documents/datas/Vascusynth_test/volumes_vascusynth"
#     gts = []
#     gts = sorted(glob(f"{img_file}/image*"))
#     # gts = sorted(glob(os.path.join(img_file,  "*.nii.gz")))
#     for i in gts:
#         print(i)
#     num_im = range(1, len(gts) + 1)
#     saved_file = "test_vascu"
#
#     file = [{"label": gt, "deco": gt, "label_art": gt} for gt in gts]
#
#     train_trans = Compose(
#         [
#             LoadImaged(keys=["deco", "label", "label_art"]),
#             ScaleIntensityd(keys=["deco", "label", "label_art"]),
#             BinaryDeconnect(keys= ["deco"], nb_deco = 20, taille_max_deco=8),
#             # BinaryDeconnect_2(keys=["deco"], nb_deco=20, taille_max_deco=8),
#             AddArtefacts(keys= ["deco", "label_art"], label = "label", mean_taches=30, seuil=0.8),
#             # AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=35, seuil=0.8),
#             AddChanneld(keys=["deco", "label", "label_art"]),
#             ToTensord(keys=["deco", "label", "label_art"]),
#         ]
#         )
#
# elif args.dataset == "cco":
#     img_file = "/home/carneiro/Documents/datas/cco/resized"
#     gts = sorted(glob(f"{img_file}/vol*"))
#     # img_file = "/home/carneiro/Documents/datas/Vascusynth_test/volumes_vascusynth"
#     # gts = sorted(glob(f"{img_file}/image*"))
#
#     # gts = sorted(glob(os.path.join(img_file,  "*.nii.gz")))
#     for i in gts:
#         print(i)
#     num_im = range(1, len(gts) + 1)
#     saved_file = "test_cco"
#
#     file = [{"label": gt, "deco": gt, "label_art": gt} for gt in gts]
#
#     train_trans = Compose(
#         [
#             LoadImaged(keys=["deco", "label", "label_art"]),
#             ScaleIntensityd(keys=["deco", "label", "label_art"]),
#             Binaries(keys=["deco", "label", "label_art"], value=0.001),
#             BinaryDeconnect(keys=["deco"], nb_deco=30, taille_max_deco=7, nb_val_ray=3),
#             AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=30, seuil=0.8),
#             # AddArtefacts(keys=["deco", "label_art"], label="label", mean_taches=35, seuil=0.8),
#             AddChanneld(keys=["deco", "label", "label_art"]),
#             ToTensord(keys=["deco", "label", "label_art"]),
#         ]
#         )
# else:
#     exit()
# ############################################ Creation des données ######################################################
# device ="cpu"


# check_ds = Dataset(data=file, transform=train_trans)
# check_loader = DataLoader(check_ds, batch_size=1, num_workers=1)
#
# # for batch_data, i in zip(check_loader, range(len(gts))):
# for batch_data, i in zip(check_loader,  range(len(gts))):
#     print(f"---------------------------traitement de l'image n°{i}-----------------------------")
#     label, deco, label_art = batch_data["label"].to(device), batch_data["deco"].to(device), batch_data["label_art"].to(device)
#     write_nifti(data=label.detach().cpu().squeeze().numpy(), file_name=f"{saved_file}/label_{i}.nii.gz", resample=False)
#     write_nifti(data=deco.detach().cpu().squeeze().numpy(), file_name=f"{saved_file}/deco_{i}.nii.gz", resample=False)
#     write_nifti(data=label_art.detach().cpu().squeeze().numpy(), file_name=f"{saved_file}/connected_art_{i}.nii.gz", resample=False)
#     pos_deco = label_art - deco
#     non_art_deco = label - pos_deco
#     write_nifti(data=pos_deco.detach().cpu().squeeze().numpy(), file_name=f"{saved_file}/pos_deco_{i}.nii.gz", resample=False)
#     write_nifti(data=non_art_deco.detach().cpu().squeeze().numpy(), file_name=f"{saved_file}/non_art_deco_{i}.nii.gz", resample=False)
#
#

########################################## Creation mask fragment #########################################################
# import nibabel as ni
# from skimage.morphology import binary_dilation, ball
# boule = ball(2)
# fragments_files = glob("test_vascu/pos_deco_*.nii.gz")
# for gt_path in fragments_files:
#     print(f"************************** {gt_path} *****************************")
#     gt = ni.load(gt_path).get_fdata()
#     gt = binary_dilation(gt, boule)
#     num = gt_path.split("/")[1]
#     write_nifti(data=gt, file_name=f"test_vascu/masked_{num}", resample=False)

import nibabel as ni
from skimage.morphology import binary_dilation, ball
boule = ball(2)
fragments_files = glob("test_brava_deco_2/pos_deco_*.nii.gz")
for gt_path in fragments_files:
    print(f"************************** {gt_path} *****************************")
    gt = ni.load(gt_path).get_fdata()
    gt = binary_dilation(gt, boule)
    num = gt_path.split("/")[1]
    write_nifti(data=gt, file_name=f"test_brava_deco_2/masked_{num}", resample=False)

###################################### generate empty patches ###############################################################"
# from personnal_transforms import generatorNoise
# import random
# n = 100
# # image = np.zeros([185, 186, 182])
# image = np.zeros([129, 129, 129])
# for i in range(n):
#     # x = random.uniform(0, 0.2)
#     # image_noisy_1 = generatorNoise(image.copy(), 30, 0.8 + x)
#     write_nifti(data=image, file_name=f"patch_vide/pos_deco_{i}.nii.gz", resample=False)

#
#   image_noisy_1 = generatorNoise(image.copy(), 30, 0.8)
# image_noisy_2= generatorNoise(image.copy(), 30, 0.8)
# image_noisy_3 = generatorNoise(image.copy(), 30, 0.8)
# image_noisy_4 = generatorNoise(image.copy(), 30, 0.8)
# image_noisy_5 = generatorNoise(image.copy(), 30, 0.8)
# write_nifti(data=image_noisy_1, file_name=f"image_129_1.nii.gz", resample=False)
# write_nifti(data=image_noisy_2, file_name=f"image_129_2.nii.gz", resample=False)
# write_nifti(data=image_noisy_3, file_name=f"image_129_3.nii.gz", resample=False)
# write_nifti(data=image_noisy_4, file_name=f"image_129_4.nii.gz", resample=False)
# write_nifti(data=image_noisy_5, file_name=f"image_129_5.nii.gz", resample=False)
