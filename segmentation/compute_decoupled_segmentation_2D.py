import numpy as np
from skimage.filters import threshold_otsu
import torch
import monai
import os
import json
import sys

sys.path.insert(0,"../sources")
import sources.grad_div_interpolation_2d as grd2D
from sources import image_utils
from sources.variational_segmentation import primal_dual_reco_chan_tv

def compute_decoupled_segmentation_2D(patient, reco_weight, name_model, max_iter = 1000):
    """
        Apply the plug and play variational 2D segmentation using the  learnt reconnecting regularization term on 2d mage.

            INPUT:
                - patient : number of the image (int)
                - reco_weight: regularization coefficient that ponderate the TV toward the other energy terms (float)
                - name_model: name of the model directory to use as a reconnecting regularization term (string)
                - max_iter: maximum iteration number of the primal dual algorithm

            OUTPUT:
                - res: 2d numpy array ; the segmentation result
        """

    # parameters for the primal dual algorithm
    lambda_n = 1
    switch_iter = 500
    sigma = 10e-3
    tau = 2/(1.1+16 * sigma)

    # charge the trained neural network with the good architecture
    model_file =f"model_reco/{name_model}/best_metric_model.pth"

    parameters_training = open(f"model_reco/{name_model}/config_training.json")
    parameters_training = json.load(parameters_training)
    norm = parameters_training["norm"]
    device = torch.device("cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=(norm)
    ).to(device)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))


    # size of the patch used for the inference of the model
    roi_size = (96, 96)

    # charge the image to segment
    if int(patient) <= 20:
        image_path = f"images/image_background_substraction/image_{patient:02d}_bg_substract_15.png"

    else:
        image_path = f"image_optimization/image_background_substraction/image_{patient:02d}_bg_substract_15.png"
    image = image_utils.read_image(image_path)
    dimy, dimx = image.shape
    image_norm = image_utils.normalize_image(image)

    # calculate the constant of the back ground and foreground constant for the data fidelity term of chan et al.
    otsu = threshold_otsu(image_norm)
    c1 = otsu
    c2 = 0

    ## gradient
    op_grady = grd2D.gradient_2d_along_axis([dimy, dimx], axis=0)
    op_gradx = grd2D.gradient_2d_along_axis([dimy, dimx], axis=1)

    L = grd2D.standard_gradient_operator_2d(op_grady, op_gradx)

    #  apply the variational segmentation with the reconnecting term plugged
    xn, __, iterations = primal_dual_reco_chan_tv(image_norm, c1, c2, L, reco_weight, switch_iter, model, roi_size, tau, sigma ,1.e-4,lambda_n, max_iter, device)
    return xn



# parameters
reco_weight = 0.012
patient = 1  # for DRIVE,number between 1 and 40 included
name_model = "07-07-2023_10-46"
max_iter = 1000

xn = compute_decoupled_segmentation_2D(patient, reco_weight, name_model, max_iter)

#save the image
file_to_save = f"results/2D/reco/optimization_{name_model}_no_fragment"
if f"optimization_{name_model}_no_fragment" not in os.listdir("results/2D/reco"):
    os.mkdir(file_to_save)

segment_8_bits_reco = ((xn >= 0.5) * 255).astype(np.uint8)
output_path_reco = f"{file_to_save}/image_{patient:02d}_reco_res_segmentation_{reco_weight:.3f}_{max_iter}.png"
image_utils.save_image(segment_8_bits_reco, output_path_reco)

