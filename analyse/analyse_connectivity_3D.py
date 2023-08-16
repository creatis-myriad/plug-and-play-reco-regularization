import numpy as np
from sources import image_utils
from monai.data import write_nifti
from glob import glob
from skimage.morphology import skeletonize, binary_dilation, ball
import nibabel as ni
import argparse
import pandas as pd
from sources.metriques import extract_skelet, calculate_rmcc, calculate_nb_composant_pourcent

def coloriage(squelette_m1, squelette_m2):

    labels = np.zeros(squelette_m1.shape)
    labels[squelette_m1 * squelette_m2 == 1] = 1  # vaisseaux en communs de M1 et M2
    labels[np.logical_and(squelette_m1 == 1, squelette_m2 == 0)] = 2  # vaisseaux de M1 mais pas M2
    labels[np.logical_and(squelette_m2 == 1, squelette_m1 == 0)] = 3  # vaisseaux M2 mais pas M1

    new_label = np.zeros(labels.shape)
    for i in range(1, 4):  # dilatation pour faire un plus rendu magueule
        matrice_confusion_i = (labels == i) * 1.0
        matrice_confusion_i = binary_dilation(matrice_confusion_i, ball(2)) * i
        new_label = np.where(matrice_confusion_i != 0, matrice_confusion_i, new_label)

    return new_label




parser = argparse.ArgumentParser(description='analyse')
parser.add_argument('dataset', type=str, default="ircad", help='dataset a utiliser')
parser.add_argument('methode', type=str, default="var", help='dataset a utiliser')
parser.add_argument('test_name', type=str, default="var", help='dataset a utiliser')

args = parser.parse_args()
test_name =args.test_name
dataset =args.dataset
methode = args.methode

reco = True
dir = True
var = True
pourcent = 0.8

if dataset == "ircad":
    num_im_max = 20
    num= np.arange(2, num_im_max +1)

elif dataset == "bullit":

    # num_bull = ['002', '003', '006', '008', '009', '010', '011', '012', '017', '018', '020', '021', '022', '023',
    #        '025', '026',
    #        '027', '033', '034', '037', '040', '042', '043', '044', '045', '060', '063', '070', '071', '074',
    #        '079', '082',
    #        '086']
    num_bull = ['002', '003', '006', '008', '009', '010', '011', '012', '017', '018', '020', '021', '022', '023', '026',
                '027', '033', '034', '037', '040', '042', '043', '044', '045', '060', '070', '071', '074', '079']
    num_im_max = len(num_bull)
    num = np.arange(0, num_im_max )
elif dataset == "bullit_rician":

    # num_bull = ['002', '003', '006', '008', '009', '010', '011', '012', '017', '018', '020', '021', '022', '023',
    #        '025', '026',
    #        '027', '033', '034', '037', '040', '042', '043', '044', '045', '060', '063', '070', '071', '074',
    #        '079', '082',
    #        '086']
    num_bull = ['002',  '003']
    num_im_max = len(num_bull)
    num = np.arange(0, num_im_max )
else:
    exit()

metrics_components = pd.DataFrame(index=["chan" , "dir", "reco"], columns=["rmcc", "nombre_composante","pourcent_recouvrement"])
d_reco = pd.DataFrame(index=num, columns=[ "rmcc", "nombre_composante","pourcent_recouvrement"])
d_chan = pd.DataFrame(index=num, columns=[ "rmcc", "nombre_composante","pourcent_recouvrement"])
d_dir = pd.DataFrame(index=num, columns=["rmcc", "nombre_composante","pourcent_recouvrement"])

for i in metrics_components.index:
    for j in metrics_components.columns:
        metrics_components.loc[i, j] = []
print(metrics_components)


for patient in num:
    print(f"********************* patient {patient} ***************************")
    try:
        if dataset == "ircad":
            image_reco_path = glob(f"results/3D/{dataset}/{methode}/{test_name}/seg_{methode}_{patient}_*")[0]
            gt_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/labels.nii.gz"
            mask_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/masks.nii.gz"

        elif dataset == "bullit":
            image_reco_path = glob(f"results/3D/{dataset}/{methode}/{test_name}/seg_{methode}_{num_bull[patient]}_*")[0]
            gt_path = f"/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/pretreated_10/Normal{num_bull[patient]}-MRA/gt.nii.gz"
            mask_path = f"/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/pretreated_10/Normal{num_bull[patient]}-MRA/mask.nii.gz"
        elif dataset == "bullit_rician":
            image_reco_path = glob(f"results/3D/{dataset}/{methode}/{test_name}/seg_{methode}_{num_bull[patient]}_*")[0]
            gt_path = f"/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/pretreated_10/Normal{num_bull[patient]}-MRA/gt.nii.gz"
            mask_path = f"/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/pretreated_10/Normal{num_bull[patient]}-MRA/mask.nii.gz"

        gt = ni.load(gt_path).get_fdata()
        gt = image_utils.normalize_image(gt)
        mask = ni.load(mask_path).get_fdata()
        mask = image_utils.normalize_image(mask)
        gt = gt * mask
        gt_skelet = image_utils.normalize_image(skeletonize((gt)))

        pretreated_gt = binary_dilation(gt_skelet, ball(2))
        print("analyse du resultat de notre approche en cours ...")
        image_reco = ni.load(image_reco_path).get_fdata()
        image_reco = (image_utils.normalize_image(image_reco) > 0.5) * 1.0
        image_reco = image_reco * mask
        image_reco_skelet = extract_skelet(image_reco, gt)
        image_reco_skelet = image_utils.normalize_image(image_reco_skelet * 1.0)
        rmcc = calculate_rmcc(image_reco, gt)
        nombre_composante, pourcent_recouvrement = calculate_nb_composant_pourcent(image_reco, gt)
        d_reco.loc[patient, "rmcc"] = round(rmcc, 4)
        d_reco.loc[patient, "nombre_composante"] = round(nombre_composante, 4)
        d_reco.loc[patient, "pourcent_recouvrement"] = round(pourcent_recouvrement, 4)
        metrics_components.loc["reco", "rmcc"].append(round(rmcc, 4))
        metrics_components.loc["reco", "pourcent_recouvrement"].append(round(pourcent_recouvrement, 4))
        metrics_components.loc["reco", "nombre_composante"].append(round(nombre_composante, 4))
    except:
        continue
metrics = metrics_components.columns.copy()
for i in metrics_components.index:
    for j in metrics:
        metrics_components.loc[i, f"std_{j}"] = np.std(metrics_components.loc[i, j])
        metrics_components.loc[i, j] = np.mean(metrics_components.loc[i, j])

print(metrics_components)

metrics_components.to_excel(f"results/3D/{dataset}/analyse_component.xlsx")
d_chan.to_excel(f"results/3D/{dataset}/component_chan.xlsx")
d_reco.to_excel(f"results/3D/{dataset}/component_reco.xlsx")
d_dir.to_excel(f"results/3D/{dataset}/component_dir.xlsx")





####################################################### 2D  ##############################################################
# chan_weight = 0.008
# reco_weight = 0.009
# dir_weight = 0.013
#
# patient_list = ["%.2d" % i for i in range(1,21)]
#
# df = pd.DataFrame(index=["chan" , "dir", "reco"], columns=["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "clDice"])
#
# for i in ["chan" , "dir", "reco"]:
#     for j in ["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "clDice"]:
#         df.loc[i, j] = []
# print(df)
#
# metrics_components_2D = pd.DataFrame(index=patient_list, columns=["1_component_var", "1_component_dec","1_component_dir","3_component_var", "3_component_dec","3_component_dir", "5_component_var", "5_component_dec", "5_component_dir", "pourcent_var",  "pourcent_dec", "pourcent_dir"])
#
# for patient in patient_list:
#     nb_pix_min = 0
#     print(f"********************* patient {patient} ***************************")
#     image_reco_path = f"results/2D/reco_res/test/image_{patient}_new_reco_segmentation_{reco_weight:.3f}_1000.png"
#     image_chan_path = f"results/2D/var/test/image_{patient}_chan_segmentation_{chan_weight:.3f}_1000.png"
#     image_dir_path = f"results/2D/dir/testPD/image_{patient}_dirPD_segmentation_{dir_weight:.3f}_1000.png"
#
#     gt_path = f"images/gt/{patient}_manual1.gif"
#     mask_path = f"images/mask_fov/{patient}_test_mask.gif"
#
#     image_reco = image_utils.normalize_image(image_utils.read_image(image_reco_path))
#     image_chan = image_utils.normalize_image(image_utils.read_image(image_chan_path))
#     image_dir = image_utils.normalize_image(image_utils.read_image(image_dir_path))
#     img_skelet = image_utils.normalize_image(skeletonize(image_reco) * 1.0)
#     img_var_skelet = image_utils.normalize_image(skeletonize(image_chan) * 1.0)
#     img_dir_skelet = image_utils.normalize_image(skeletonize(image_dir) * 1.0)
#
#     mask = image_utils.normalize_image(image_utils.read_image(mask_path))
#     gt = image_utils.normalize_image(image_utils.read_image(gt_path))
#     gt_skelet = image_utils.normalize_image(skeletonize(gt) * 1.0)
#
#
#     val_decoupled, __, __ = val_pixels_per_component(img_skelet)
#     val_var, __, __ = val_pixels_per_component(img_var_skelet)
#     val_dir, __, __ = val_pixels_per_component(img_dir_skelet)
#
#     val_gt, __, __ = val_pixels_per_component(gt_skelet)
#     pourcent = 0.8
#
#
#     value_1_stat, value_3_stat, value_5_stat, n_pourcent = get_componants_stats(val_gt, val_var, pourcent)
#
#     print(f" pourcent 1 compenent var : {value_1_stat} \n pourcent 3 compenent var : {value_3_stat} \n pourcent 5 compenent var : {value_5_stat} \n 80% de la gt : {n_pourcent} ")
#
#     metrics_components.loc[patient,"1_component_var"] = round(value_1_stat, 4)
#     metrics_components.loc[patient,"3_component_var"] = round(value_3_stat, 4)
#     metrics_components.loc[patient,"5_component_var"] = round(value_5_stat, 4)
#     metrics_components.loc[patient,"pourcent_var"] = n_pourcent
#
#     value_1_stat, value_3_stat, value_5_stat, n_pourcent = get_componants_stats(val_gt, val_decoupled, pourcent)
#     metrics_components.loc[patient,"1_component_dec"] = round(value_1_stat, 4)
#     metrics_components.loc[patient,"3_component_dec"] = round(value_3_stat, 4)
#     metrics_components.loc[patient,"5_component_dec"] = round(value_5_stat, 4)
#     metrics_components.loc[patient,"pourcent_dec"] = n_pourcent
#     print(f" pourcent 1 compenent decoupled : {value_1_stat} \n pourcent 3 compenent decoupled : {value_3_stat} \n pourcent 5 decoupled var : {value_5_stat} \n 80% de la gt : {n_pourcent} ")
#     value_1_stat, value_3_stat, value_5_stat, n_pourcent = get_componants_stats(val_gt, val_dir, pourcent)
#     metrics_components.loc[patient, "1_component_dir"] = round(value_1_stat, 4)
#     metrics_components.loc[patient, "3_component_dir"] = round(value_3_stat, 4)
#     metrics_components.loc[patient, "5_component_dir"] = round(value_5_stat, 4)
#     metrics_components.loc[patient, "pourcent_dir"] = n_pourcent
#
#     metrics_components.to_excel(f"results/2D/analyse_components.xlsx")
#
#     print(f" pourcent 1 compenent dir : {value_1_stat} \n pourcent 3 compenent dir : {value_3_stat} \n pourcent 5 dir : {value_5_stat} \n 80% de la gt : {n_pourcent} ")


# # taille_bin = 50
#
# #################################### Histogrammes magueule ##########################################################
# # # tout sur le meme plot
# plt.figure()
# bins = range(0, max(val_decoupled + val_var + val_gt )+taille_bin, taille_bin)  # pour faire les barres
# plt.hist([val_decoupled, val_var, val_gt],  color=["green", "red", "blue"], bins=bins,histtype = 'bar', alpha=0.3, log=True, label=["approche découplée", "approche variationnelle", "groundtruth"]) #on se tape des barres ici
# plt.xlabel("nombre de pixel")
# plt.ylabel("nombre de composantes connexes")
# plt.title("Approche variationnelle")
# plt.legend()
# plt.grid()
# plt.savefig(f"{file_to_save}/histo_val_decoupled_var.png")
#
# histo_best_component(histo_decoupled, histo_var, histo_gt, 10)
# plt.savefig(f"{file_to_save}/10_first_composant_vessel.png", format = "png")