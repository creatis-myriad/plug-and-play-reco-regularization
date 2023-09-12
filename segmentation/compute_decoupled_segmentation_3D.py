import numpy as np
from monai.data import  write_nifti
import monai
import torch
import os
import nibabel as ni
import json
import sys

sys.path.insert(0,"../sources")
from sources import image_utils
import sources.grad_div_interpolation_3d as grd3D
from sources.variational_segmentation_3D import primal_dual_ind_reconnect_3D_no_frag


def compute_decoupled_segmentation_3D(patient, reco_weight, name_model, max_iter = 1000):
    """
       Apply the plug and play variational 3D segmentation using the  learnt reconnecting regularization term on 2d mage.

           INPUT:
               - patient : number of the image (int)
               - reco_weight: regularization coefficient that ponderate the TV toward the other energy terms (float)
               - name_model: name of the model directory to use as a reconnecting regularization term (string)
               - max_iter: maximum iteration number of the primal dual algorithm

           OUTPUT:
               - res: 2d numpy array ; the segmentation result
       """


    # parameters for the primal dual algorithm
    sigma= 10e-3
    tau = 2/(1.1 + 24 * sigma)
    lambda_n = 1
    switch_iter = 500
    data_weight =1


    # charge the trained neural network with the good architecture
    file_training=f"model_reco_3D/{name_model}/best_metric_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters_training = open(f"model_reco_3D/{name_model}/config_training.json")
    parameters_training = json.load(parameters_training)
    norm = parameters_training["norm"]
    size_patch = parameters_training["patch_size"]
    roi_size = size_patch
    print(roi_size)
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm = (norm),
    ).to(device)

    if device == "cuda":
        model.load_state_dict(torch.load(file_training))
    else:
        model.load_state_dict(torch.load(file_training, map_location=torch.device('cpu')))


    # charge the image to segment
    image_path = f"../../datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/preprocessed.nii.gz"
    gt_path = f"../../datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/labels.nii.gz"
    mask_path = f"../../datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/masks.nii.gz"
    if f"seg_reco_{patient}_{reco_weight:.5f}.nii.gz" in os.listdir(directory_to_save):
        print("already done !")
        exit()

    gt = ni.load(gt_path).get_fdata()
    gt = (image_utils.normalize_image(gt) * 255).astype(np.uint8)
    gt_norm = image_utils.normalize_image(gt)

    image = ni.load(image_path).get_fdata()
    image_norm = image_utils.normalize_image(image)

    mask = ni.load(mask_path).get_fdata()
    mask = image_utils.normalize_image(mask)

    # calculate the constant of the back ground and foreground constant for the data fidelity term of chan et al.
    c1 = (np.sum(image_norm * (gt_norm == 1)) / np.sum(gt_norm == 1)) + 0.05
    c2 = (np.sum(image_norm * (gt_norm == 0)) / np.sum(gt_norm == 0))

    dimz, dimy, dimx = image.shape
    #  gradient
    op_gradz = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=0)
    op_grady = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=1)
    op_gradx = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=2)

    L = grd3D.standard_gradient_operator_3d(op_gradz, op_grady, op_gradx)

    #  apply the variational segmentation with the reconnecting term plugged
    xn, __, iterations = primal_dual_ind_reconnect_3D_no_frag(image_norm, gt_norm, mask, c1, c2, L, reco_weight, data_weight,
                                                          model, roi_size, switch_iter, tau, sigma, 1.e-5, lambda_n,
                                                          max_iter, device)

    return xn


# parameters
reco_weight = 0.012
patient = 1  #for Ircad ,number between 1 and 20 included
name_model = "07-07-2023_10-46"
max_iter = 1000

xn = compute_decoupled_segmentation_3D(patient, reco_weight, name_model, max_iter)


directory_to_save = f"results/3D/ircad/reco/optimization_training_{name_model}"

if f"optimization_training_{name_model}" not in os.listdir(f"results/3D/ircad/reco"):
    os.mkdir(directory_to_save)
    segment_reconnector = (xn >= 0.5) * 1.0

    write_nifti(data=(segment_reconnector * 255).astype(np.uint8),
                file_name=f"{directory_to_save}/seg_reco_{patient}_{reco_weight:.5f}.nii.gz", resample=False)
    print("done !!!!")