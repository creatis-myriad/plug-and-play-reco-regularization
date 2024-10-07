import sources.grad_div_interpolation_3d as grd3D
import monai
import nibabel as ni
from sources import image_utils
import json
import numpy as np
import torch
from source_3D.post_treatement import monai_predict_image

def primal_dual_ind_reconnect_3D(image, c1, c2, L, chan_weight, switch_iter, model, roi_size, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100, device=torch.device("cpu")):
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
    xn = np.zeros(image.shape, np.float64)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float64)
    energy = 100
    nb_iter = 0
    grad_h = ((c1 - image) ** 2 - (c2 - image) ** 2)
    iterations = []

    while ((energy > epsilon or nb_iter < switch_iter) and nb_iter < max_iter):
        old_xn = xn.copy()
        nb_iter += 1
        if nb_iter < switch_iter:
            pn = xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
            pn = proj(pn)
        else:
            pn = xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
            reconnections = monai_predict_image(proj(pn), model, roi_size, device=device)
            pn = reconnections.copy()
            pn = proj(pn)
        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        qn = qn - sigma*proxg(qn/sigma, chan_weight, 1/sigma)
        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)
        energy = np.linalg.norm(xn.reshape(-1) - old_xn.reshape(-1), 2)
        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    return xn, nb_iter, iterations

def proj(image):
    image[image < 0] = 0
    image[image > 1] = 1
    return image

def proxg(u, chan_weight, gamma):
    vx = u[:int(u.shape[0] / 3)]
    vy = u[int(u.shape[0] / 3):int(2*(u.shape[0] / 3))]
    vz = u[int(2 * (u.shape[0] / 3)):]

    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    norm = np.tile(norm, 3)
    prox_norm = (1 - (gamma*chan_weight / np.maximum(norm, gamma*chan_weight)))*u
    return prox_norm



def reconnector_plug_and_play(image_path, gt_path, mask_path, tv_weight, model_directory_path, switch_iter=500, sigma=10e-3, max_iter=1000, lambda_n=1):
    """
        :param image_path: path toward the source_2D to segment
        :param gt_path:  path toward the groundtruth
        :param mask_path:  path toward the mask
        :param tv_weight: regularisation coefficient linked to the total variation
        :param model_directory_path: path to the directory containing the reconnecting model
        :param switch_iter: iteration from which we inject our reconnecting model
        :param sigma: optimisation gradient step.
        :param max_iter: maximal number of iteration possible
        :param lambda_n: relaxation parameter
        return
        """
    file_training = f"{model_directory_path}/best_metric_model.pth"
    tau = 2 / (1.1 + 24 * sigma)

    gt = ni.load(gt_path).get_fdata()
    gt = (image_utils.normalize_image(gt) * 255).astype(np.uint8)
    gt_norm = image_utils.normalize_image(gt)

    image = ni.load(image_path).get_fdata()
    image_norm = image_utils.normalize_image(image)

    mask = ni.load(mask_path).get_fdata()
    mask = image_utils.normalize_image(mask)


    # calculate C1 et c2 knowing the groundtruth to optimize the result for each source_2D
    c1 = (np.sum(image_norm * (gt_norm == 1)) / np.sum(gt_norm == 1)) + 0.05
    c2 = (np.sum(image_norm * (gt_norm == 0)) / np.sum(gt_norm == 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters_training = open(f"{model_directory_path}/config_training.json")
    parameters_training = json.load(parameters_training)
    norm = parameters_training["norm"]
    roi_size = parameters_training["patch_size"]

    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=(norm),
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

    xn_reconnector, __, __ = primal_dual_ind_reconnect_3D(image_norm, c1, c2, L, tv_weight,
                                                      model, roi_size, switch_iter, tau, sigma, 1.e-5, lambda_n,
                                                      max_iter, device)

    segment_reconnector = (xn_reconnector >= 0.5) * 1.0
    return segment_reconnector
