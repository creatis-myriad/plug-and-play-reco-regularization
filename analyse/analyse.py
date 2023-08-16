from sources import image_utils
import nibabel as ni
from glob import glob
import pandas as pd

import os
from sources import metriques
from sources import nifti_image



########################### rename the file ####################################
# # copy the input data in the good file for the segmentation with nnUnet
# input_directory = "/home/carneiro/Documents/Tours/Documents/Master/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task504_BullitTest_crop/imagesTs"
# origin_input_directory = "/home/carneiro/Documents/Tours/Documents/datas/Bullit_iso/Bullit_V2/"
#
# mask_files = sorted(glob(os.path.join(origin_input_directory, "Nor*/dataIso*")))
# num_bull = ['002', '003', '006', '008', '009', '010', '011', '012', '017', '018', '020', '021', '022', '023', '025', '026',
#        '027', '033', '034', '037', '040', '042', '043', '044', '045', '060', '063', '070', '071', '074', '079', '082', '086']
#
# for a, i in enumerate(mask_files):
#     print(i)
#     im = nifti_image.read_nifti(i)
#     name = i.split("/")[-1].split(".")[0]
#     nifti_image.save_nifti(im, f"{input_directory}/{name}_{num_bull[a]}_0000.nii.gz", metadata_model = i)
#
# exit()

# ##### test nnUnet sur Bullit #######
# segmentation_directory = "/home/carneiro/Documents/Master/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task504_BullitTest_crop_no_postprocess/segmentationsTs"
# label_directory = "/home/carneiro/Documents/Master/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task504_BullitTest_crop_no_postprocess/labelsTs"
# label_directory_2 = "/home/carneiro/Documents/datas/Bullit_iso/Bullit_V2/"
#
# num_bull = ['002', '003', '006', '008', '009', '010', '011', '012', '017', '018', '020', '021', '022', '023', '025', '026',
#        '027', '033', '034', '037', '040', '042', '043', '044', '045', '060', '063', '070', '071', '074', '079', '082', '086']
#
# seg_files = sorted(glob(os.path.join(segmentation_directory, "dat*")))
# lab_files_1 = sorted(glob(os.path.join(label_directory, "bin*")))
# # lab_files_2 = sorted(glob(os.path.join(label_directory_2, "Nor*/bin*")))
# mask_files = sorted(glob(os.path.join(label_directory_2, "Nor*/brainMaskIso*")))
#
# for i, j in zip(seg_files, lab_files_1):
#     print(i, j)
# # for i, k in zip(seg_files, lab_files_2):
# #     print(i, k)
#
# df = pd.DataFrame(index=num_bull, columns=["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "clDice"])
# # for i in ["chan" , "dir", "reco"]:
# #     for j in ["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "clDice"]:
# #         df.loc[i, j] = []
# # print(df)
#
#
# for i, nn_unet_path, gt_path, mask_path, in zip(range(len(num_bull)), seg_files, lab_files_1, mask_files):
#
#     gt = ni.load(gt_path).get_fdata()
#     gt = image_utils.normalize_image(gt)
#
#     mask = ni.load(mask_path).get_fdata()
#     mask = image_utils.normalize_image(mask)
#     gt = gt * mask
#
#
#     image = ni.load(nn_unet_path).get_fdata()
#     image = (image_utils.normalize_image(image) > 0.5) * 1.0
#     image = image * mask
#
#     acc_chan, tpr_chan, tnr_chan = image_utils.compute_accuracy(image, gt, mask)
#     print(f"*accuracy done *")
#     df.loc[num_bull[i], "acc"] = (acc_chan)
#     df.loc[num_bull[i], "sp"] = (tnr_chan)
#     df.loc[num_bull[i], "se"] = (tpr_chan)
#     mcc_chan = image_utils.compute_mcc(image, gt, mask)
#     df.loc[num_bull[i], "mcc"] = (mcc_chan)
#     dice_chan = image_utils.compute_dice(image, gt, mask)
#     print(f"*dice done *")
#     df.loc[num_bull[i], "dice"] = (dice_chan)
#     __, __, cal_chan = metriques.cldice(image, gt)
#     df.loc[num_bull[i], "clDice"] = (cal_chan)
#     print(f"*cldice done *")
#     print(df)
#
# df.to_excel(f"nnUnet_bullit_cropped_no_post_process.xlsx")


# #### test nnUnet sur Brava #######
# segmentation_directory = "/home/carneiro/Documents/Master/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Brava_nnUnet_output"
# label_directory = "/home/carneiro/Documents/Master/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Brava/labelsTr"
# num_brava = [1, 3, 4, 5, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
#              38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58]
#
# seg_files = sorted(glob(os.path.join(segmentation_directory, "BRAVA*")))
# lab_files_1 = sorted(glob(os.path.join(label_directory, "MRA*")))
#
# for i, j in zip(seg_files, lab_files_1):
#     print(i, j)
# exit()
# df = pd.DataFrame(index=num_brava, columns=["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "clDice"])
# # for i in ["chan" , "dir", "reco"]:
# #     for j in ["acc" , "sp", "se", "mcc", "dice","ov", "c", "a", "l", "cal", "clDice"]:
# #         df.loc[i, j] = []
# # print(df)
#
#
# for i, nn_unet_path, gt_path, mask_path, in zip(range(len(num_brava)), seg_files, lab_files_1):
#
#     gt = ni.load(gt_path).get_fdata()
#     gt = image_utils.normalize_image(gt)
#
#
#     acc_chan, tpr_chan, tnr_chan = image_utils.compute_accuracy(image, gt)
#     print(f"*accuracy done *")
#     df.loc[num_bull[i], "acc"] = (acc_chan)
#     df.loc[num_bull[i], "sp"] = (tnr_chan)
#     df.loc[num_bull[i], "se"] = (tpr_chan)
#     mcc_chan = image_utils.compute_mcc(image, gt, mask)
#     df.loc[num_bull[i], "mcc"] = (mcc_chan)
#     dice_chan = image_utils.compute_dice(image, gt, mask)
#     print(f"*dice done *")
#     df.loc[num_bull[i], "dice"] = (dice_chan)
#     __, __, cal_chan = metriques.cldice(image, gt)
#     df.loc[num_bull[i], "clDice"] = (cal_chan)
#     print(f"*cldice done *")
#     print(df)
#
# df.to_excel(f"nnUnet_bullit.xlsx")


###### remove header to compare with other results ######################
segmentation_directory = "/home/carneiro/Documents/Master/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task504_BullitTest_crop_no_postprocess/segmentationsTs"
seg_files = sorted(glob(os.path.join(segmentation_directory, "dat*")))
save_file = "results/3D/bullit/nnUNet/"
for i in seg_files:
    seg = nifti_image.read_nifti(i)
    name = i.split("/")[-1]
    nifti_image.save_nifti(seg, save_file+name)