import numpy as np
from sources import image_utils
from monai.data import write_nifti
from glob import glob
import nibabel as ni
import argparse
import pandas as pd
from sources.metriques import euler_number_error_numpy, b0_error_numpy, b1_error_numpy, b2_error_numpy
from skimage.morphology import remove_small_objects
import os
parser = argparse.ArgumentParser(description='analyse')
parser.add_argument('dataset', type=str, default="ircad", help='dataset a utiliser')
parser.add_argument('type_methode', type=str, default="reco", help='dataset a utiliser')
parser.add_argument('chemin', type=str, default="ircad", help='dataset a utiliser')

args = parser.parse_args()

dataset = args.dataset
methode = args.type_methode
chemin = args.chemin
if methode == "reco":
    nom = chemin.split('/')[-1]
    nom = nom[14:]
else:
    nom = methode

if dataset == "ircad":
    num_im_max = 20
    num= np.arange(2, num_im_max +1)

# a fixer magueule
min_size = 90
if "ugly_betty.xlsx" in os.listdir(f"/home/carneiro/Documents/Master/myPrimalDual/results/3D/{dataset}"):
    df = pd.read_excel(f"/home/carneiro/Documents/Master/myPrimalDual/results/3D/{dataset}/ugly_betty.xlsx")
    columns = []
    indexs = []
    for i in df.columns[1:]:
        columns.append(i)
    for j in df["Unnamed: 0"]:
        indexs.append(j)
    print()
    df_new = pd.DataFrame(df.values[:, 1:], index=indexs, columns=columns)
    df = df_new

else:
    df = pd.DataFrame(index=[nom, "gt"], columns=["b0", "b0_error", "b1","b1_error","b2", "b2_error", "euler", "euler_error"])

for j in ["b0", "b0_error", "b1","b1_error","b2", "b2_error", "euler", "euler_error"]:
    df.loc[nom, j] = []

d_gt = pd.DataFrame(index=num, columns=["b0", "b0_error", "b1","b1_error","b2", "b2_error", "euler", "euler_error"])
d_reco = pd.DataFrame(index=num, columns=["b0", "b0_error", "b1","b1_error","b2", "b2_error", "euler", "euler_error"])


for j in df.columns:
    df.loc[nom, j] = []
    df.loc["gt", j] = []
print(df)

for patient in num:
    print(f"********************* patient {patient} ***************************")
    image_path = glob(f"{chemin}/seg_{methode}_{patient}_*")[0]
    gt_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/labels.nii.gz"
    mask_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/masks.nii.gz"

    gt = ni.load(gt_path).get_fdata()
    gt = image_utils.normalize_image(gt)
    mask = ni.load(mask_path).get_fdata()
    mask = image_utils.normalize_image(mask)
    gt = gt * mask
    # gt = remove_small_objects(gt > 0.5, min_size=min_size)

    __, euler_number_true, __ = euler_number_error_numpy(gt, gt)
    __, b0_number_true, __ = b0_error_numpy(gt, gt)
    __, b1_number_true, __ = b1_error_numpy(gt, gt)
    __, b2_number_true, __ = b2_error_numpy(gt, gt)

    d_gt.loc[patient, "b0"] = round(b0_number_true, 4)
    d_gt.loc[patient, "b1"] = round(b1_number_true, 4)
    d_gt.loc[patient, "b2"] = round(b2_number_true, 4)
    d_gt.loc[patient, "euler"] = round(euler_number_true, 4)
    df.loc["gt", "b0"].append(round(b0_number_true, 4))
    df.loc["gt", "b1"].append(round(b1_number_true, 4))
    df.loc["gt", "b2"].append(round(b2_number_true, 4))
    df.loc["gt", "euler"].append(round(euler_number_true, 4))

    image_reco = ni.load(image_path).get_fdata()
    image_reco = (image_utils.normalize_image(image_reco) > 0.5) * 1.0
    image_reco = image_reco * mask
    write_nifti(data=image_reco, file_name=f"results/test_small_object/reco_{patient}.nii.gz", resample=False)
    image_reco = remove_small_objects(image_reco > 0.5, min_size=min_size)
    # image_reco = remove_small_holes(image_reco > 0.5, area_threshold=30, connectivity=1)

    write_nifti(data=image_reco,
                file_name=f"results/test_small_object/reco_{patient}_{min_size}.nii.gz", resample=False)
    euler_number_error, __, euler_number_pred = euler_number_error_numpy(gt, image_reco)
    b0_number_error, __, b0_number_pred = b0_error_numpy(gt, image_reco)
    b1_number_error, __, b1_number_pred = b1_error_numpy(gt, image_reco)
    b2_number_error, __, b2_number_pred = b2_error_numpy(gt, image_reco)

    d_reco.loc[patient, "b0"] = round(b0_number_pred, 4)
    d_reco.loc[patient, "b1"] = round(b1_number_pred, 4)
    d_reco.loc[patient, "b2"] = round(b2_number_pred, 4)
    d_reco.loc[patient, "euler"] = round(euler_number_pred, 4)

    d_reco.loc[patient, "b0_error"] = round(b0_number_error, 4)
    d_reco.loc[patient, "b1_error"] = round(b1_number_error, 4)
    d_reco.loc[patient, "b2_error"] = round(b2_number_error, 4)
    d_reco.loc[patient, "euler_error"] = round(euler_number_error, 4)

    df.loc[nom, "b0"].append(round(b0_number_pred, 4))
    df.loc[nom, "b1"].append(round(b1_number_pred, 4))
    df.loc[nom, "b2"].append(round(b2_number_pred, 4))
    df.loc[nom, "euler"].append(round(euler_number_pred, 4))

    df.loc[nom, "b0_error"].append(round(b0_number_error, 4))
    df.loc[nom, "b1_error"].append(round(b1_number_error, 4))
    df.loc[nom, "b2_error"].append(round(b2_number_error, 4))
    df.loc[nom, "euler_error"].append(round(euler_number_error, 4))


metrics = ["b0", "b0_error", "b1","b1_error","b2", "b2_error", "euler", "euler_error"]
for i in [nom, "gt"]:
    for j in metrics:
        df.loc[i, f"std_{j}"] = np.std(df.loc[i, j])
        df.loc[i, j] = np.mean(df.loc[i, j])

df.to_excel(f"results/3D/{dataset}/ugly_betty.xlsx")
d_reco.to_excel(f"results/3D/{dataset}/ugly_betty_{nom}.xlsx")
d_gt.to_excel(f"results/3D/{dataset}/ugly_betty_gt.xlsx")
