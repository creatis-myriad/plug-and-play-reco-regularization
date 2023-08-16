import numpy as np
import sources.grad_div_interpolation_3d as grd3D
from monai.data import write_nifti
from sources.variational_segmentation_3D import primal_dual_ind_chan_tv_3D
from sources import image_utils
import nibabel as ni

def compute_variational_segmentation_3D(patient, chan_weight):
    #parameters independant to the image
    sigma= 10e-3
    tau = 2/(1.1+24 * sigma)
    lambda_n = 1
    max_iter = 1000

    # paths to the images
    image_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/preprocessed.nii.gz"
    gt_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/labels.nii.gz"

    gt = ni.load(gt_path).get_fdata()
    gt = (image_utils.normalize_image(gt) * 255).astype(np.uint8)
    gt_norm = image_utils.normalize_image(gt)

    image = ni.load(image_path).get_fdata()
    image_norm = image_utils.normalize_image(image)

    # calcul de C1 et c2 en connaissant la vérité terrain pour optimiser les résultats pour chaque image
    c1 = (np.sum(image_norm * (gt_norm == 1)) / np.sum(gt_norm == 1)) + 0.05
    c2 = (np.sum(image_norm * (gt_norm == 0)) / np.sum(gt_norm == 0))

    dimz, dimy, dimx = image_norm.shape
    #parameters
    ## gradient
    op_gradz = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=0)
    op_grady = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=1)
    op_gradx = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=2)
    L = grd3D.standard_gradient_operator_3d(op_gradz, op_grady, op_gradx)

    # Segmentation
    xn, __ = primal_dual_ind_chan_tv_3D(image_norm, c1, c2, L,chan_weight, tau, sigma, 1.e-5,lambda_n, max_iter)
    segment_chan = (xn >= 0.5) * 255

    write_nifti(data=segment_chan.astype(np.uint8), file_name=f"results/3D/ircad/var/optimization/seg_var_{patient}_{chan_weight:.5f}.nii.gz", resample=False)
