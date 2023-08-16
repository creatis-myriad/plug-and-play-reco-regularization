import numpy as np
from monai.data import write_nifti
import pandas as pd
import nibabel as ni
from sources.quantificateurSegmentation import evaluate_image_binaire3D
from sources import image_utils
import os

import argparse

parser = argparse.ArgumentParser(description='analyse')
parser.add_argument('dataset', type=str, default="ircad", help='dataset a utiliser')
parser.add_argument('methode', type=str, default="var", help='dataset a utiliser')
parser.add_argument('optimisation_name', type=str, default="var", help='dataset a utiliser')

args = parser.parse_args()
optimisation_name =args.optimisation_name
dataset =args.dataset
methode = args.methode

num_im_max = 20
num = np.arange(1, num_im_max +1)
chan_weights = np.concatenate((np.array([0.000001, 0.00001,0.00005, 0.0001, 0.0005]),np.arange(0.001, 0.065 + 0.001, 0.001)))
main_file = f"/home/carneiro/Documents/Master/myPrimalDual/results/3D/{dataset}/{methode}/{optimisation_name}"

if methode == "var" or methode == "dir":
    name_analyse_file = 'analyse_optimization'
    name_test_file = "test"
    main_file = f"/home/carneiro/Documents/Master/myPrimalDual/results/3D/{dataset}/{methode}/optimization"
else:
    name_analyse_file = f"analyse_optimization_training_{optimisation_name[len('optimization_training_'):]}"
    name_test_file = f"test_training_{optimisation_name[len('optimization_training_'):]}"
test_file = f"results/3D/{dataset}/{methode}/{name_test_file}"
analyse_file = f"results/3D/{dataset}/{methode}/{name_analyse_file}"

if name_analyse_file not in os.listdir( f"results/3D/{dataset}/{methode}"):
    os.mkdir(analyse_file)
if name_test_file not in os.listdir(f"results/3D/{dataset}/{methode}"):
    os.mkdir(test_file)

glob_result = pd.DataFrame(index= range(0, num_im_max), columns=[f"{methode}_weight_mcc",f"{methode}_weight_dice", f"{methode}_weight_acc",  "mcc","dice", "acc"])

for num_im in range(0,num_im_max):
    acc_max = 0
    mcc_max = 0
    chan_weight_mcc = 0
    chan_weight_acc = 0
    dice_max = 0
    chan_weight_dice = 0
    if dataset == "ircad":
        gt_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{num[num_im]}/labels.nii.gz"
        mask_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{num[num_im]}/masks.nii.gz"
    else:
        gt_path = f"/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/pretreated_10/Normal{num[num_im]}-MRA/gt.nii.gz"
        mask_path = f"/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/pretreated_10/Normal{num[num_im]}-MRA/mask.nii.gz"
    gt = ni.load(gt_path).get_fdata()
    gt = image_utils.normalize_image(gt)
    mask = ni.load(mask_path).get_fdata()
    mask = image_utils.normalize_image(mask)
    gt = gt * mask
    df = pd.DataFrame(index=chan_weights, columns=["acc_mean", "spe_mean", "dice_mean", "mcc_mean"])
    image_opt = np.zeros(gt.shape)
    for i in chan_weights:
        print(f"{main_file}/seg_{methode}_{num[num_im]}_{i:.5f}.nii.gz")
        seg = f"{main_file}/seg_{methode}_{num[num_im]}_{i:.5f}.nii.gz"
        try:
            image = ni.load(seg).get_fdata()
            image = image_utils.normalize_image(image)
            print(image.shape, gt.shape, mask.shape)

            image = image * mask
            image = (image > 0.5) * 1.0

            metrics = evaluate_image_binaire3D(image, gt, mask)
            result_dice = metrics["dice"]
            result_mcc = metrics["mcc"]
            accuracy = metrics["acc"]
            spe = metrics["sp"]

            df.loc[i,"acc_mean"] = accuracy
            df.loc[i,"spe_mean"] = spe
            df.loc[i,"dice_mean"] = result_dice
            df.loc[i,"mcc_mean"] = result_mcc
            # print("mcc", result_mcc)
            if result_mcc > mcc_max:
                mcc_max = result_mcc
                chan_weight_mcc = i
                image_opt = image.copy()
            if accuracy > acc_max:
                acc_max = accuracy
                chan_weight_acc = i
            if result_dice > dice_max:
                dice_max = result_dice
                chan_weight_dice = i
            df.to_excel( f"{analyse_file}/analyse_final_{methode}_patient_{num[num_im]}.xlsx")
        except:
            continue
    glob_result.loc[num_im,"acc"] = acc_max
    glob_result.loc[num_im,f"{methode}_weight_mcc"] = chan_weight_mcc
    glob_result.loc[num_im,f"{methode}_weight_acc"] = chan_weight_acc
    glob_result.loc[num_im,"mcc"] = mcc_max
    glob_result.loc[num_im,f"{methode}_weight_dice"] = chan_weight_dice
    glob_result.loc[num_im,"dice"] = dice_max
    write_nifti(data=(image_opt * 255).astype(np.uint8), file_name=f"{test_file}/seg_{methode}_{num[num_im]}_{chan_weight_mcc:.5f}.nii.gz",
                resample=False)
    glob_result.to_excel(f"{analyse_file}/analyse_globale.xlsx")