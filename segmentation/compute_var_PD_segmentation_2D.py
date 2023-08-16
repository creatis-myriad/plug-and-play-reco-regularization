import numpy as np
import sources.grad_div_interpolation_2d as grd2D
from sources import image_utils
from sources.variational_segmentation import primal_dual_ind_chan_tv
from skimage.filters import threshold_otsu

def compute_variational_segmentation_3D(patient, chan_weight):
    lambda_n = 1
    max_iter = 1000
    sigma = 10e-3
    tau = 2/(1.1+16 * sigma)

    if int(patient) <= 20:
        image_path = f"images/image_background_substraction/image_{patient:02d}_bg_substract_15.png"
    else:
        image_path = f"image_optimization/image_background_substraction/image_{patient:02d}_bg_substract_15.png"

    file_to_save = "results/2D/var/optimization"
    image = image_utils.read_image(image_path)

    dimy, dimx = image.shape
    image_norm = image_utils.normalize_image(image)
    otsu = threshold_otsu(image_norm)

    #parameters
    c1 = otsu
    c2 = 0

    ## gradient
    op_grady = grd2D.gradient_2d_along_axis([dimy, dimx], axis=0)
    op_gradx = grd2D.gradient_2d_along_axis([dimy, dimx], axis=1)

    L = grd2D.standard_gradient_operator_2d(op_grady, op_gradx)

    xn = primal_dual_ind_chan_tv(image_norm, c1, c2, L,chan_weight, tau, sigma ,1.e-4,lambda_n, max_iter)
    segment_8_bits_chan = ((xn>=0.5) * 255).astype(np.uint8)
    output_path_chan = f"{file_to_save}/image_{patient:02d}_var_segmentation_{chan_weight:.3f}_{max_iter}.png"
    image_utils.save_image(segment_8_bits_chan, output_path_chan)

