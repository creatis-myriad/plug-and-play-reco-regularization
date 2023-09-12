from monai.data import write_nifti
import numpy as np
import nibabel as ni

import sys
sys.path.insert(0,"../sources")
import sources.grad_div_interpolation_3d as grd3D
from sources.variational_segmentation_3D import primal_dual_ind_chan_tv_3D
from sources import image_utils


def compute_variational_segmentation_3D(patient, chan_weight, max_iter=1000):
    """
            Apply the variational 3D segmentation using the TV regularization term on 2d mage.

                INPUT:
                    - patient : number of the image (int)
                    - chan_weight: regularization coefficient that ponderate the TV toward the other energy terms (float)
                    - max_iter: maximum iteration number of the primal dual algorithm (int)

                OUTPUT:
                    - res:3d numpy array ; the segmentation result
            """

    # parameters for the primal dual algorithm
    sigma= 10e-3
    tau = 2/(1.1+24 * sigma)
    lambda_n = 1

    # paths to the images
    image_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/preprocessed.nii.gz"
    gt_path = f"/home/carneiro/Documents/datas/ircad_iso_V3/pretreated_ircad_10/3Dircadb1.{patient}/labels.nii.gz"

    gt = ni.load(gt_path).get_fdata()
    gt = (image_utils.normalize_image(gt) * 255).astype(np.uint8)
    gt_norm = image_utils.normalize_image(gt)

    image = ni.load(image_path).get_fdata()
    image_norm = image_utils.normalize_image(image)

    # calculate the constant of the back ground and foreground constant for the data fidelity term of chan et al.
    c1 = (np.sum(image_norm * (gt_norm == 1)) / np.sum(gt_norm == 1)) + 0.05
    c2 = (np.sum(image_norm * (gt_norm == 0)) / np.sum(gt_norm == 0))

    dimz, dimy, dimx = image_norm.shape

    # gradient
    op_gradz = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=0)
    op_grady = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=1)
    op_gradx = grd3D.gradient_3d_along_axis([dimz, dimy, dimx], axis=2)
    L = grd3D.standard_gradient_operator_3d(op_gradz, op_grady, op_gradx)

    # apply variational Segmentation
    xn, __ = primal_dual_ind_chan_tv_3D(image_norm, c1, c2, L,chan_weight, tau, sigma, 1.e-5,lambda_n, max_iter)

    return xn


patient = 1
chan_weight = 0.001
max_iter = 1000

xn = compute_variational_segmentation_3D(patient, chan_weight, max_iter)

segment_chan = (xn >= 0.5) * 255
write_nifti(data=segment_chan.astype(np.uint8), file_name=f"results/3D/ircad/var/optimization/seg_var_{patient}_{chan_weight:.5f}.nii.gz", resample=False)
