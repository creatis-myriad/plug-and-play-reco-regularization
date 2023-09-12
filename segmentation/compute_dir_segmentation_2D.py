import numpy as np
from skimage.filters import threshold_otsu
import sys

sys.path.insert(0,"../sources")
import sources.grad_div_interpolation_2d as grd2D
from sources import image_utils
from sources.variational_segmentation import primal_dual_ind_chan_tv

def compute_directional_segmentation_2D(patient, dir_weight, max_iter=1000):
    """
    Apply the variational 2D segmentation using the directionnal TV regularization term on 2d mage.

        INPUT:
            - patient : number of the image (int)
            - dir_weight: regularization coefficient that ponderate the TV toward the other energy terms (float)
            - max_iter: maximum iteration number of the primal dual algorithm (int)

        OUTPUT:
            - res: 2d numpy array ; the segmentation result
    """

    # parameters for the primal dual algorithm
    lambda_n = 1
    sigma= 10e-3
    tau = 2/(1.1+16 * sigma)

    # charge images and estimated directions of blood vessels thanks to RORPO
    if int(patient) <= 20:
        image_path = f"images/image_background_substraction/image_{patient:02d}_bg_substract_15.png"
        rorpo_path = f"images/rorpo/RORPO_{patient:02d}_10_2_5_0"
    else:
        image_path = f"image_optimization/image_background_substraction/image_{patient:02d}_bg_substract_15.png"
        rorpo_path = f"image_optimization/rorpo/RORPO_{patient:02d}_10_2_5_0"
    file_to_save = "results/2D/dir/optimizationPD"
    rorpo = image_utils.read_image(f"{rorpo_path}.png")
    rorpo_vx = image_utils.read_image(f"{rorpo_path}_dirx.tif")
    rorpo_vy = image_utils.read_image(f"{rorpo_path}_diry.tif")

    rorpo_threshold = (rorpo > 15).astype(np.uint8)

    rorpo_vx = rorpo_vx * rorpo_threshold
    rorpo_vy = rorpo_vy * rorpo_threshold

    ori = np.stack((rorpo_vy, rorpo_vx))

    image = image_utils.read_image(image_path)
    image_norm = image_utils.normalize_image(image)
    otsu = threshold_otsu(image_norm)

    # calculate the constant of the back ground and foreground constant for the data fidelity term of chan et al.
    c1 = otsu
    c2 = 0

    op_gradx = grd2D.gradient_2d_along_axis_anisotropy_correction(ori, axis=1)
    op_grady = grd2D.gradient_2d_along_axis_anisotropy_correction(ori, axis=0)
    # Directional gradient
    op_grad_dir = grd2D.directional_2d_gradient(ori)

    # Mixed gradient
    L = grd2D.mixed_gradient_operator_2d(op_grady, op_gradx, op_grad_dir)

    # apply the directional segmentation
    xn, __ = primal_dual_ind_chan_tv(image_norm, c1, c2, L, dir_weight, tau, sigma, 1.e-4, lambda_n, max_iter)

    return xn

patient = 1
dir_weight = 0.01
max_iter = 1000
file_to_save = "results/2D/dir/optimizationPD"

xn = compute_directional_segmentation_2D(patient, dir_weight, max_iter)

segment_8_bits_dir = ((xn>=0.5) * 255).astype(np.uint8)
output_path_dir = f"{file_to_save}/image_{patient:02d}_dir_segmentation_{dir_weight:.3f}_{max_iter}.png"
image_utils.save_image(segment_8_bits_dir, output_path_dir)