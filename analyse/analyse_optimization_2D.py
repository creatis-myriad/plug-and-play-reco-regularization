from sources import image_utils
from sources.quantificateurSegmentation import evaluate_image_binaire
import numpy as np
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser(description='analyse')
parser.add_argument('methode', type=str, default="var", help='dataset a utiliser')
parser.add_argument('optimisation_name', type=str, default="var", help='dataset a utiliser')

args = parser.parse_args()
optimisation_name =args.optimisation_name
methode = args.methode

file_to_analyse = f"results/2D/{methode}/{optimisation_name}"
name_test_file = f"test_{optimisation_name}"
name_analyse_file = f"analyse_{optimisation_name}"
test_file = f"results/2D/{methode}/{name_test_file}"
analyse_file = f"results/2D/{methode}/{name_analyse_file}"
if name_analyse_file not in os.listdir( f"results/2D/{methode}"):
    os.mkdir(analyse_file)
if name_test_file not in os.listdir(f"results/2D/{methode}"):
    os.mkdir(test_file)

patient_list = ["%.2d" % i for i in range(1, 41)]
chan_weights = np.arange(0.001, 0.051, 0.001)
glob_result = pd.DataFrame(index= patient_list, columns=[f"{methode}_weight_mcc",f"{methode}_weight_dice", f"{methode}_weight_acc",  "mcc","dice", "acc"])
df = pd.DataFrame(index=chan_weights, columns=["acc_mean", "spe_mean", "dice_mean", "mcc_mean"])


for patient in patient_list:
    acc_max = 0
    mcc_max = 0
    chan_weight_mcc = 0
    chan_weight_acc = 0
    dice_max = 0
    chan_weight_dice = 0
    if int(patient) <= 20:
        mask_path = f"images/mask_fov/{patient}_test_mask.gif"
        gt_path = f"images/gt/{patient}_manual1.gif"

    else:
        mask_path = f"image_optimization/mask_fov/{patient}_training_mask.gif"
        gt_path = f"image_optimization/gt/{patient}_manual1.gif"
    gt = image_utils.read_image(gt_path) / 255
    mask = image_utils.read_image(mask_path)
    for weight in chan_weights:
        image_path = f"{file_to_analyse}/image_{patient}_{methode}_segmentation_{weight:.3f}_1000.png"

        image = image_utils.read_image(image_path)
        image = image_utils.normalize_image(image)


        dico = evaluate_image_binaire(image, gt, mask)

        df.loc[weight, "acc_mean"] = dico["acc"]
        df.loc[weight, "spe_mean"] = dico["sp"]
        df.loc[weight, "dice_mean"] = dico["dice"]
        df.loc[weight, "mcc_mean"] = dico["mcc"]

        if dico["mcc"] > mcc_max:
            mcc_max = dico["mcc"]
            chan_weight_mcc = weight
            image_opt = image.copy()
        if dico["acc"] > acc_max:
            acc_max = dico["acc"]
            chan_weight_acc = weight
        if dico["dice"] > dice_max:
            dice_max =  dico["dice"]
            chan_weight_dice = weight
    glob_result.loc[patient, "acc"] = acc_max
    glob_result.loc[patient, f"{methode}_weight_mcc"] = chan_weight_mcc
    glob_result.loc[patient, f"{methode}_weight_acc"] = chan_weight_acc
    glob_result.loc[patient, "mcc"] = mcc_max
    glob_result.loc[patient, f"{methode}_weight_dice"] = chan_weight_dice
    glob_result.loc[patient, "dice"] = dice_max
    segment_8_bits_reco = ((image_opt >= 0.5) * 255).astype(np.uint8)
    output_path_reco = f"{test_file}/image_{patient}_{methode}_segmentation_{chan_weight_mcc:.3f}_1000.png"
    image_utils.save_image(segment_8_bits_reco, output_path_reco)
    df.to_excel(f"{analyse_file}/analyse_{patient}.xlsx")
    glob_result.to_excel(f"{analyse_file}/analyse_glob.xlsx")
    print(df)


