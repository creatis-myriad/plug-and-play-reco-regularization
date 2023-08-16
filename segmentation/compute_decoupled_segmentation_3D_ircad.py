import numpy as np
import sources.grad_div_interpolation_3d as grd3D
from monai.data import  write_nifti
from sources.variational_segmentation_3D import primal_dual_ind_reconnect_3D_no_frag
import monai
import torch
import os
import nibabel as ni
from sources import image_utils
import json

def compute_decoupled_segmentation_3D(num_image, reco_weight, nom_training, max_iter = 1000):
    #parameters independant to the image
    sigma= 10e-3
    tau = 2/(1.1 + 24 * sigma)
    lambda_n = 1
    switch_iter = 500
    data_weight =1
    file_training=f"model_reco_3D/{nom_training}/best_metric_model.pth"

    #creation du fichier si besoin
    directory_to_save = f"results/3D/ircad/reco/optimization_training_{nom_training}"

    if f"optimization_training_{nom_training}" not in os.listdir(f"results/3D/ircad/reco"):
        os.mkdir(directory_to_save)

    image_path = f"../../datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{num_image}/preprocessed.nii.gz"
    gt_path = f"../../datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{num_image}/labels.nii.gz"
    mask_path = f"../../datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{num_image}/masks.nii.gz"
    if f"seg_reco_{num_image}_{reco_weight:.5f}.nii.gz" in os.listdir(directory_to_save):
        print("already done !")
        exit()

    gt = ni.load(gt_path).get_fdata()
    gt = (image_utils.normalize_image(gt) * 255).astype(np.uint8)
    gt_norm = image_utils.normalize_image(gt)

    image = ni.load(image_path).get_fdata()
    image_norm = image_utils.normalize_image(image)

    mask = ni.load(mask_path).get_fdata()
    mask = image_utils.normalize_image(mask)

    # calcul de C1 et c2 en connaissant la vérité terrain pour optimiser les résultats pour chaque image
    c1 = (np.sum(image_norm * (gt_norm == 1)) / np.sum(gt_norm == 1)) + 0.05
    c2 = (np.sum(image_norm * (gt_norm == 0)) / np.sum(gt_norm == 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters_training = open(f"model_reco_3D/{nom_training}/config_training.json")
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

    dimz, dimy, dimx = image.shape
    ## gradient
    op_gradz = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=0)
    op_grady = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=1)
    op_gradx = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=2)

    L = grd3D.standard_gradient_operator_3d(op_gradz, op_grady, op_gradx)

    xn_reconnector, __, iterations = primal_dual_ind_reconnect_3D_no_frag(image_norm, gt_norm, mask, c1, c2, L, reco_weight, data_weight,
                                                          model, roi_size, switch_iter, tau, sigma, 1.e-5, lambda_n,
                                                          max_iter, device)

    segment_reconnector = (xn_reconnector >= 0.5) * 1.0


    write_nifti(data=(segment_reconnector * 255).astype(np.uint8),
                    file_name=f"{directory_to_save}/seg_reco_{num_image}_{reco_weight:.5f}.nii.gz", resample=False)
    print("done !!!!")