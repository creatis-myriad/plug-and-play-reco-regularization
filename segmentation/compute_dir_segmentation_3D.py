import numpy as np
from sources import image_utils
import nibabel as ni
from glob import glob
import sources.grad_div_interpolation_3d as grd3D
from sources.variational_segmentation_3D import primal_dual_dir_tv_3D
from monai.data import write_nifti

def compute_directional_segmentation_3D(patient, dir_weight):
    sigma= 10e-3
    tau = 2/(1.1+24 * sigma)
    lambda_n = 1
    max_iter = 1000

    file_to_save = f"results/3D/ircad/dir/optimization"

    ############## paths to the images
    # paths to the images
    image_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/preprocessed.nii.gz"
    gt_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/labels.nii.gz"
    mask_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/masks.nii.gz"
    rorpo_intensity_path = glob(f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/*intensity.nii.gz")[0]
    rorpo_dirx_path = glob(f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/*dirx.nii.gz")[0]
    rorpo_diry_path = glob(f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/*diry.nii.gz")[0]
    rorpo_dirz_path = glob(f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/*dirz.nii.gz")[0]


    gt = ni.load(gt_path).get_fdata()
    gt = (image_utils.normalize_image(gt) * 255).astype(np.uint8)
    gt_norm = image_utils.normalize_image(gt)

    image = ni.load(image_path).get_fdata()
    image_norm = image_utils.normalize_image(image)

    mask = ni.load(mask_path).get_fdata()
    mask = image_utils.normalize_image(mask)


    rorpo = ni.load(rorpo_intensity_path).get_fdata()
    rorpo = (image_utils.normalize_image(rorpo) * 255).astype(np.uint8)
    mcc, treshold = image_utils.compute_best_mcc(rorpo, gt, mask)
    rorpo_vx = ni.load(rorpo_dirx_path).get_fdata()

    rorpo_vy = ni.load(rorpo_diry_path).get_fdata()

    rorpo_vz = ni.load(rorpo_dirz_path).get_fdata()

    # calcul de C1 et c2 en connaissant la vérité terrain pour optimiser les résultats pour chaque image
    c1 = (np.sum(image_norm * (gt_norm == 1)) / np.sum(gt_norm == 1)) + 0.05
    c2 = (np.sum(image_norm * (gt_norm == 0)) / np.sum(gt_norm == 0))

    rorpo_threshold = rorpo > treshold

    rorpo_vx = rorpo_vx * rorpo_threshold
    rorpo_vy = rorpo_vy * rorpo_threshold
    rorpo_vz = rorpo_vz * rorpo_threshold

    ori = np.stack((rorpo_vz, rorpo_vy, rorpo_vx))

    # Compute the matrix formulation of the directional gradient
    # Isotropic standard gradient
    op_gradx = grd3D.gradient_3d_along_axis_anisotropy_correction(ori, axis=2)
    op_grady = grd3D.gradient_3d_along_axis_anisotropy_correction(ori, axis=1)
    op_gradz = grd3D.gradient_3d_along_axis_anisotropy_correction(ori, axis=0)

    # Directional gradient
    op_grad_dir = grd3D.directional_3d_gradient(ori)
    # Mixed gradient
    L = grd3D.mixed_gradient_operator_3d(op_gradz, op_grady, op_gradx, op_grad_dir)
    xn, __ = primal_dual_dir_tv_3D(image_norm, c1, c2, L,dir_weight, tau, sigma, 1.e-5,lambda_n, max_iter)

    segment_8_bits_reco = ((xn >= 0.5) * 255).astype(np.uint8)
    write_nifti(data=segment_8_bits_reco, file_name=f"{file_to_save}/seg_dir_{patient}_{dir_weight:.5f}.nii.gz", resample=False)

