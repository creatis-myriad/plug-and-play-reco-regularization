import numpy as np
import sources.grad_div_interpolation_2d as grd2D
from sources import image_utils
from sources.variational_segmentation import primal_dual_reco_chan_tv
from skimage.filters import threshold_otsu
import torch
import monai
import os
import json

def compute_decoupled_segmentation_2D(patient, reco_weight, path_model, max_iter = 1000):
    lambda_n = 1
    switch_iter = 500
    sigma = 10e-3
    tau = 2/(1.1+16 * sigma)

    model_file =f"model_reco/{path_model}/best_metric_model.pth"

    parameters_training = open(f"model_reco/{path_model}/config_training.json")
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

    roi_size = (96, 96)

    name_model = path_model
    if int(patient) <= 20:
        image_path = f"images/image_background_substraction/image_{patient:02d}_bg_substract_15.png"

    else:
        image_path = f"image_optimization/image_background_substraction/image_{patient:02d}_bg_substract_15.png"

    file_to_save = f"results/2D/reco_res/optimization_{name_model}_no_fragment"
    if f"optimization_{name_model}_no_fragment" not in os.listdir("results/2D/reco_res"):
        os.mkdir(file_to_save)

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

    xn2, __, iterations = primal_dual_reco_chan_tv(image_norm, c1, c2, L, reco_weight, switch_iter, model, roi_size, tau, sigma ,1.e-4,lambda_n, max_iter, device)

    segment_8_bits_reco = ((xn2 >= 0.5) * 255).astype(np.uint8)
    output_path_reco = f"{file_to_save}/image_{patient:02d}_reco_res_segmentation_{reco_weight:.3f}_{max_iter}.png"
    image_utils.save_image(segment_8_bits_reco, output_path_reco)

