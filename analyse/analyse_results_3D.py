import pandas as pd
import numpy as np
from sources import image_utils
from sources import metriques
from glob import glob
import nibabel as ni
import os
import argparse


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
    num = np.arange(2, num_im_max +1)

if "analyse_final.xlsx" in os.listdir(f"/home/carneiro/Documents/Master/myPrimalDual/results/3D/{dataset}"):
    df = pd.read_excel(f"/home/carneiro/Documents/Master/myPrimalDual/results/3D/{dataset}/analyse_final.xlsx")
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
    df = pd.DataFrame(index=[nom], columns=["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "t_sens", "t_prec", "clDice"])

for j in ["acc", "sp", "se", "mcc", "dice", "ov", "c", "a", "l", "cal", "t_sens", "t_prec", "clDice"]:
    df.loc[nom, j] = []

d = pd.DataFrame(index=num, columns=["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "t_sens", "t_prec", "clDice"])


for patient in num:
    print(f"********************* patient {patient} ***************************")
    if methode == "var":
        image_path = glob(f"{chemin}/seg_{methode}_{patient}_*")[0]
    else:
        image_path = glob(f"{chemin}/seg_{methode}_{patient}_*")[0]
    gt_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/labels.nii.gz"
    mask_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/masks.nii.gz"


    gt = ni.load(gt_path).get_fdata()
    gt = image_utils.normalize_image(gt)
    mask = ni.load(mask_path).get_fdata()
    mask = image_utils.normalize_image(mask)
    gt = gt * mask

    print("analyse du resultat de notre approche en cours ...")
    image = ni.load(image_path).get_fdata()
    image = (image_utils.normalize_image(image) > 0.5) * 1.0
    image = image * mask
    print("reco size", image.shape)

    # metriques
    acc, tpr, tnr = image_utils.compute_accuracy(image, gt, mask)
    mcc = image_utils.compute_mcc(image, gt, mask)
    dice = image_utils.compute_dice(image, gt, mask)
    t_sens, t_prec, cldice = metriques.cldice(image, gt)


    # saisie des resultats
    df.loc[nom, "acc"].append(acc)
    df.loc[nom, "sp"].append(tnr)
    df.loc[nom, "se"].append(tpr)
    df.loc[nom, "mcc"].append(mcc)
    df.loc[nom, "dice"].append(dice)
    df.loc[nom, "clDice"].append(cldice)
    df.loc[nom, "t_sens"].append(t_sens)
    df.loc[nom, "t_prec"].append(t_prec)

    d.loc[patient, "sp"] = tnr
    d.loc[patient, "acc"] = acc
    d.loc[patient, "se"] = tpr
    d.loc[patient, "dice"] = dice
    d.loc[patient, "mcc"] = mcc
    d.loc[patient, "clDice"] = cldice
    d.loc[patient, "t_sens"] = t_sens
    d.loc[patient, "t_prec"] = t_prec

for j in ["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "t_sens", "t_prec", "clDice"]:
    df.loc[nom, f"std_{j}"] = np.std(df.loc[nom, j])
    df.loc[nom, j] = np.mean(df.loc[nom, j])

df.to_excel(f"results/3D/{dataset}/analyse_final.xlsx")
d.to_excel(f"results/3D/{dataset}/usuelles_{nom}.xlsx")
