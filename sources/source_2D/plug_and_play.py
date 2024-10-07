import numpy as np
import sources.source_2D.grad_div_interpolation_2d as grd2D
from sources import image_utils
from skimage.filters import threshold_otsu
import scipy
import torch
import monai
import json
from sources.source_2D.post_treatement import monai_predict_image

def proj(image):
    image[image < 0] = 0
    image[image > 1] = 1
    return image

def proxg(u, chan_weight, gamma):
    vx = u[:int(u.shape[0] / 2)]
    vy = u[int(u.shape[0] / 2):]
    norm = np.sqrt(vx * vx + vy * vy)
    norm = np.tile(norm, 2)
    prox_norm = (1 - (gamma*chan_weight / np.maximum(norm, gamma*chan_weight)))*u
    return prox_norm


def primal_dual_reco_chan_tv(image, c1, c2, L,chan_weight, switch_iter , model, roi_size, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5, max_iter=100, device="cpu"):
    """
    :param image: source_2D to segment
    :param c1: foreground constant of the chan model
    :param c2: background constant of the chan model
    :param L: gradient operator
    :param chan_weight: regularisation coefficient that weight the total variation
    :param switch_iter: iteration from which we inject our reconnecting model
    :param model: reconnecting model that has been previously trained
    :param roi_size: size of the patch used during the training of the model
    :param tau: optimisation gradient step.
    :param sigma: optimisation gradient step.
    :param epsilon: threshold that permits to say if the algorithm converged or not
    :param lambda_n: relaxation parameter
    :param max_iter: maximal number of iteration possible
    :param device: cpu or gpu
    return the segmented source_2D, the number of iteration made, evolution of the segmented source_2D through the optimization scheme
    """
    xn = np.zeros(image.shape,np.float64)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float64)
    energy = 100
    nb_iter = 0
    grad_h = ((c1 - image) ** 2 - (c2 - image) ** 2)
    iterations = []
    while ((energy > epsilon or nb_iter < switch_iter) and nb_iter < max_iter):
        old_xn = xn.copy()
        nb_iter += 1
        pn = xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
        if nb_iter < switch_iter:
            pn = proj(pn)
        else:
            pn = proj(monai_predict_image(proj(pn), model, roi_size, device=device))
        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        qn = qn - sigma * proxg(qn / sigma, chan_weight, 1 / sigma)
        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)
        energy = np.linalg.norm(xn - old_xn, 2)
        iterations.append(xn.copy())
        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    return xn, nb_iter, iterations

def reconnector_plug_and_play(image_path, tv_weight, model_directory_path, switch_iter=500, sigma=10e-3, max_iter=1000, lambda_n=1):
    """
    :param image_path: source_2D path toward the source_2D to segment
    :param tv_weight: regularisation coefficient linked to the total variation
    :param model_directory_path: path to the directory containing the reconnecting model
    :param switch_iter: iteration from which we inject our reconnecting model
    :param sigma: optimisation gradient step.
    :param max_iter: maximal number of iteration possible
    :param lambda_n: relaxation parameter
    return
    """

    tau = 2/(1.1+16 * sigma)

    model_file =f"{model_directory_path}/best_metric_model.pth"

    parameters_training = open(f"{model_directory_path}/config_training.json")
    parameters_training = json.load(parameters_training)
    norm = parameters_training["norm"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=(norm)
    ).to(device)


    if device == "cuda":
        model.load_state_dict(torch.load(model_file)).to(device)
    else:
        model.load_state_dict(torch.load(model_file, map_location="cpu"))

    roi_size = parameters_training["roi_size"]

    image = image_utils.read_image(image_path)

    dimy, dimx = image.shape
    image_norm = image_utils.normalize_image(image, 1)
    otsu = threshold_otsu(image_norm)

    #parameters
    c1 = otsu
    c2 = 0

    ## gradient
    op_grady = grd2D.gradient_2d_along_axis([dimy, dimx], axis=0)
    op_gradx = grd2D.gradient_2d_along_axis([dimy, dimx], axis=1)

    L = grd2D.standard_gradient_operator_2d(op_grady, op_gradx)
    print(c1, c2, L, tv_weight, tau, sigma, 1.e-4, lambda_n, max_iter)
    xn2, diff_list, iterations = primal_dual_reco_chan_tv(image_norm, c1, c2, L, tv_weight, switch_iter, model, roi_size, tau, sigma, 1.e-4,lambda_n, max_iter, device)

    segment_8_bits_reco = ((xn2 >= 0.5) * 255).astype(np.uint8)

    return segment_8_bits_reco

