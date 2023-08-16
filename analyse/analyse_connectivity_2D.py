import pandas as pd
import sys
import numpy as np
from sources import image_utils
from sources import metriques
import os
from glob import glob
from skimage.morphology import remove_small_objects, remove_small_holes


from sources.metriques import euler_number_error_numpy, b0_error_numpy_2D, b1_error_numpy_2D



nom = "var"
dir_to_analyse = "results/2D/var/test_optimization"


nom = "dir"
dir_to_analyse = "results/2D/dir/test_optimizationPD"

nom = "8_reco"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-22-2023_17-13_no_seuil_no_fragment"
# #
nom = "8_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-22-2023_18-05_no_seuil_no_fragment"
# #
nom = "12_reco"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-23-2023_11-57_no_seuil_no_fragment"
# #
nom = "12_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-23-2023_11-08_no_seuil_no_fragment"
# #
nom = "17_reco"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-13-2023_11-10_no_seuil_no_fragment"
# #
nom = "17_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-15-2023_15-58_no_seuil_no_fragment"
# #
# #
nom = "all_reco_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-29-2023_10-01_no_seuil_no_fragment"
# #
#
nom = "12_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_03-31-2023_17-57_no_seuil_no_fragment"


nom = "dir_reco_8"
dir_to_analyse = "results/2D/dir_reco/test_optimization_03-22-2023_18-05"

nom = "dir_reco_12"
dir_to_analyse = "results/2D/dir_reco/test_optimizationPD"

nom = "dir_reco_17"
dir_to_analyse = "results/2D/dir_reco/test_optimization_03-15-2023_15-58"
# #
nom = "reco_frag_12"
dir_to_analyse = "results/2D/reco_res/test_optimization_04-04-2023_11-01_no_seuil_fragment"
#
nom = "post_treatement_var"
dir_to_analyse = "results/2D/var/test_optimization/post_treatement_nn/"

nom = "reco_frag_all_++"
dir_to_analyse = "results/2D/reco_res/test_optimization_04-27-2023_16-57_fragment"

nom = "reco_PD_batch_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-16-2023_15-59_no_fragment"
nom_im = "reco"

nom = "reco_PD_instance_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-16-2023_16-08_no_fragment"
nom_im = "reco"



nom = "reco_simple_PD_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-21-2023_12-51_no_fragment"
nom_im = "reco"
# #
nom = "denoise_PD_instance_1000"
dir_to_analyse = "results/2D/reco_res/test_optimization_05-21-2023_12-53_no_fragment"
nom_im = "reco"

nom = "batchnorm_PD_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_07-07-2023_10-46_no_fragment"
nom_im = "reco"
# #
nom = "batchnorm_PD_reconnect"
dir_to_analyse = "results/2D/reco_res/test_optimization_07-07-2023_15-55_no_fragment"
nom_im = "reco"
nom = "batchnorm_PD_reconnect_denoise"
dir_to_analyse = "results/2D/reco_res/test_optimization_07-07-2023_13-20_no_fragment"
nom_im = "reco"


patient_list = ["%.2d" % i for i in range(1,41)]
min_size = 20


if "connectivity.xlsx" in os.listdir(f"/home/carneiro/Documents/Master/myPrimalDual/results/2D"):
    df = pd.read_excel(f"/home/carneiro/Documents/Master/myPrimalDual/results/2D/connectivity.xlsx")
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
    df = pd.DataFrame(index=["gt", nom], columns=["clDice", "b0", "b1", "euler", "b0_error", "b1_error", "euler_error"])

for j in df.columns:
    df.loc[nom, j] = []
    df.loc["gt", j] = []


# d = pd.DataFrame(index=patient_list, columns=["acc" , "sp", "se", "mcc", "dice", "clDice"])


for patient in patient_list:
    print(f"********************* patient {patient} ***************************")
    image_chan_path = glob(f"{dir_to_analyse}/image_{patient}_*_10*.png")[0]

    if int(patient) <= 20:
        mask_path = f"images/mask_fov/{patient}_test_mask.gif"
        gt_path = f"images/gt/{patient}_manual1.gif"

    else:
        mask_path = f"image_optimization/mask_fov/{patient}_training_mask.gif"
        gt_path = f"image_optimization/gt/{patient}_manual1.gif"

    image_chan = image_utils.normalize_image(image_utils.read_image(image_chan_path))
    mask = image_utils.normalize_image(image_utils.read_image(mask_path))
    gt = image_utils.normalize_image(image_utils.read_image(gt_path))
    image_chan = image_chan * mask

    __, __, cal_chan = metriques.cldice(image_chan, gt)
    df.loc[nom, "clDice"].append(cal_chan)

    image_chan = remove_small_objects(image_chan > 0.5, min_size=min_size, connectivity=2)
    image_chan = remove_small_holes(image_chan > 0.5, area_threshold=10, connectivity=1)

    name_image = image_chan_path.split("/")[-1]

    if "post_treatement" not in os.listdir(dir_to_analyse):
        os.mkdir(f"{dir_to_analyse}/post_treatement")

    segment_8_bits_chan = ((image_chan >= 0.5) * 255).astype(np.uint8)
    output_path_reco = f"{dir_to_analyse}/post_treatement/{name_image}"
    image_utils.save_image(segment_8_bits_chan, output_path_reco)


    gt = remove_small_objects(gt > 0.5, min_size=min_size, connectivity=2)
    gt = remove_small_holes (gt > 0.5, area_threshold=10, connectivity=1)

    new_gt_path = f"{gt_path.split('/')[0]}/{gt_path.split('/')[1]}"
    if "post_treatement" not in os.listdir(new_gt_path):
        os.mkdir(f"{new_gt_path}/post_treatement")

    segment_8_bits_chan = ((gt >= 0.5) * 255).astype(np.uint8)
    output_path_reco = f"{new_gt_path}/post_treatement/{gt_path.split('/')[-1]}"
    image_utils.save_image(segment_8_bits_chan, output_path_reco)

    # print(f"*cldice done *")
    euler_number_error, euler_number_true, euler_number_pred = euler_number_error_numpy(gt, image_chan)
    b0_number_error, b0_number_true, b0_number_pred = b0_error_numpy_2D(gt, image_chan)
    b1_number_error, b1_number_true, b1_number_pred = b1_error_numpy_2D(gt, image_chan)

    df.loc["gt", "b0"].append(round(b0_number_true, 4))
    df.loc["gt", "b1"].append(round(b1_number_true, 4))
    df.loc["gt", "euler"].append(round(euler_number_true, 4))
    print(f" euler estime : {b0_number_pred - b1_number_pred} ,  euler calc : {euler_number_pred}")
    print(f" euler estime : {b0_number_true - b1_number_true} ,  euler calc : {euler_number_true}")

    df.loc[nom, "b0"].append(round(b0_number_pred, 4))
    df.loc[nom, "b1"].append(round(b1_number_pred, 4))
    df.loc[nom, "euler"].append(round(euler_number_pred, 4))

    df.loc[nom, "b0_error"].append(round(b0_number_error, 4))
    df.loc[nom, "b1_error"].append(round(b1_number_error, 4))
    df.loc[nom, "euler_error"].append(round(euler_number_error, 4))

columns = ["clDice", "b0", "b1", "euler", "b0_error", "b1_error", "euler_error"]
for j in columns:
    df.loc[nom, f"std_{j}"] = round(np.std(df.loc[nom, j]),3)
    df.loc[nom, j] = round(np.mean(df.loc[nom, j]), 3)
    df.loc["gt", f"std_{j}"] = round(np.std(df.loc["gt", j]), 3)
    df.loc["gt", j] = round(np.mean(df.loc["gt", j]), 3)

df.to_excel(f"results/2D/connectivity.xlsx")
