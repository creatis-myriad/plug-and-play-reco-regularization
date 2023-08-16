import pandas as pd
import sys
import numpy as np
from sources import image_utils
from sources import metriques
import os
from glob import glob

nom = "var"
dir_to_analyse = "results/2D/var/test_optimization"

# #
nom = "dir"
dir_to_analyse = "results/2D/dir/test_optimizationPD"
# # # #
nom = "8_reco"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-22-2023_17-13_no_seuil_no_fragment"
# # # #
nom = "8_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-22-2023_18-05_no_seuil_no_fragment"
# # #
nom = "12_reco"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-23-2023_11-57_no_seuil_no_fragment"
# # #
# # #
nom = "12_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-23-2023_11-08_no_seuil_no_fragment"
# # #
nom = "17_reco"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-13-2023_11-10_no_seuil_no_fragment"
# #
# #
nom = "17_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-15-2023_15-58_no_seuil_no_fragment"
# min_size = 20
#
nom = "all_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-29-2023_10-01_no_seuil_no_fragment"
nom_im = "reco"


nom = "12_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-31-2023_17-57_no_seuil_no_fragment"
nom_im = "reco"
# min_size = 20

nom = "dir_reco_8"
dir_to_analyse = "results/2D/dir_reco/test_optimization_03-22-2023_18-05"
nom_im = "dir_reco"

nom = "dir_reco_12"
dir_to_analyse = "results/2D/dir_reco/test_optimizationPD"
nom_im = "dir_reco"

nom = "dir_reco_17"
dir_to_analyse = "results/2D/dir_reco/test_optimization_03-15-2023_15-58"
nom_im = "dir_reco"

nom = "reco_frag_12"
dir_to_analyse = "results/2D/reco_res/test_optimization_04-04-2023_11-01_no_seuil_fragment"
nom_im = "reco"


nom = "post_treatement_var"
dir_to_analyse = "results/2D/var/test_optimization/post_treatement/"
nom_im = "post_treatement_var"

nom = "reco_frag_all_++"
dir_to_analyse = "results/2D/reco_res/test_optimization_04-27-2023_16-57_fragment"
nom_im = "reco"

nom = "reco_PD_instance"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-16-2023_11-15_no_fragment"
nom_im = "reco"

nom = "reco_PD_batch"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-16-2023_11-24_no_fragment"
nom_im = "reco"

nom = "reco_PD_batch_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-16-2023_15-59_no_fragment"
nom_im = "reco"
#
nom = "reco_PD_instance_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-16-2023_16-08_no_fragment"
nom_im = "reco"
patient_list = ["%.2d" % i for i in range(1,41)]


nom = "reco_simple_PD_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-21-2023_12-51_no_fragment"
nom_im = "reco"
#
nom = "denoise_PD_instance_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-21-2023_12-53_no_fragment"
nom_im = "reco"


nom = "batchnorm_PD_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_07-07-2023_10-46_no_fragment"
nom_im = "reco"
#
nom = "batchnorm_PD_reconnect"
dir_to_analyse = "results/2D/reco_res/test_optimization_07-07-2023_15-55_no_fragment"
nom_im = "reco"
nom = "batchnorm_PD_reconnect_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_07-07-2023_13-20_no_fragment"
nom_im = "reco"


patient_list = ["%.2d" % i for i in range(1,41)]
if "analyse_final.xlsx" in os.listdir(f"/home/carneiro/Documents/Master/myPrimalDual/results/2D"):
    df = pd.read_excel(f"/home/carneiro/Documents/Master/myPrimalDual/results/2D/analyse_final.xlsx")
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
    df = pd.DataFrame(index=[nom], columns=["acc" , "sp", "se", "mcc", "dice","ov", "clDice"])

for j in df.columns:
    df.loc[nom, j] = []

d = pd.DataFrame(index=patient_list, columns=["acc" , "sp", "se", "mcc", "dice", "clDice"])

dir_visu = f"{dir_to_analyse}/visu"
if "visu" not in os.listdir(dir_to_analyse):
    os.mkdir(f"{dir_to_analyse}/visu")

for patient in patient_list:
    print(f"********************* patient {patient} ***************************")
    image_chan_path = glob(f"{dir_to_analyse}/image_{patient}_*_10*.png")[0]
    if int(patient) <= 20:
        mask_path = f"images/mask_fov/{patient}_test_mask.gif"
        gt_path = f"images/gt/{patient}_manual1.gif"

    else:
        mask_path = f"image_optimization/mask_fov/{patient}_training_mask.gif"
        gt_path = f"image_optimization/gt/{patient}_manual1.gif"


    image_chan = (image_utils.normalize_image(image_utils.read_image(image_chan_path))>0.5)*1.0

    mask = image_utils.normalize_image(image_utils.read_image(mask_path))
    gt = image_utils.normalize_image(image_utils.read_image(gt_path))

    # image_chan = remove_small_objects(image_chan > 0.5, min_size=min_size, connectivity=2)
    # gt = remove_small_objects(gt > 0.5, min_size=min_size, connectivity=2)

    acc_chan, tpr_chan, tnr_chan = image_utils.compute_accuracy(image_chan, gt, mask)

    print(f"*accuracy done *")


    df.loc[nom, "acc"].append(acc_chan)
    df.loc[nom, "sp"].append(tnr_chan)


    df.loc[nom, "se"].append(tpr_chan)

    mcc_chan = image_utils.compute_mcc(image_chan, gt, mask)
    print(f"*mcc done *")

    df.loc[nom, "mcc"].append(mcc_chan)


    dice_chan = image_utils.compute_dice(image_chan, gt, mask)

    print(f"*dice done *")

    df.loc[nom, "dice"].append(dice_chan)


    __, __, cal_chan = metriques.cldice(image_chan, gt)
    df.loc[nom, "clDice"].append(cal_chan)

    print(f"*cldice done *")

    # print("visu de merde")
    white = np.logical_and(image_chan != 0, gt != 0)
    green = np.logical_and(image_chan == 0, gt != 0)
    red = np.logical_and(image_chan != 0, gt == 0)
    blue = white.copy()
    green = np.logical_or(green, white)
    red = np.logical_or(red, white)
    image_confusion = np.array([red, green, blue])
    image_confusion = ((np.dstack((red,green, blue)) >= 0.5) * 255).astype(np.uint8)
    output_path_reco = f"{dir_visu}/image_{patient}_{nom_im}_segmentation.png"
    image_utils.save_image(image_confusion, output_path_reco)
for j in ["acc" , "sp", "se", "mcc", "dice","ov", "clDice"]:
    df.loc[nom, f"std_{j}"] = round(np.std(df.loc[nom, j]), 3)
    df.loc[nom, j] = round(np.mean(df.loc[nom, j]), 3)

df.to_excel(f"results/2D/analyse_final.xlsx")
