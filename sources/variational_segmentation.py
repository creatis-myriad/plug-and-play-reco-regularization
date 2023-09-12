import numpy as np
import torch
from monai.inferers import sliding_window_inference
import math

def primal_dual_ind_chan_tv(image, c1, c2, L,chan_weight, tau  = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100):

    xn = np.zeros(image.shape, np.float64)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float64)
    energy = 100

    nb_iter = 0


    grad_h = ((c1 - image) ** 2 - (c2 - image) ** 2)

    while (energy > epsilon and nb_iter < max_iter):
        old_xn = xn.copy()
        nb_iter += 1
        pn =xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
        pn = proj(pn)
        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        qn = qn - sigma*proxg(qn/sigma, chan_weight, 1/sigma)
        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)
        energy = np.linalg.norm(xn - old_xn, 2)

        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    return xn

def primal_dual_reco_chan_tv(image, c1, c2, L,chan_weight, switch_iter , model, roi_size, tau  = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100, device="cpu"):

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


def primal_dual_reco_res_chan_tv(image, c1, c2, L,chan_weight, switch_iter , model, roi_size, tau  = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100, device="cpu"):
    results_reco = []
    xn = np.zeros(image.shape, np.float64)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float64)
    energy = 100
    nb_iter = 0
    grad_h = ((c1 - image) ** 2 - (c2 - image) ** 2)
    while ((energy > epsilon or nb_iter < switch_iter) and nb_iter < max_iter):
        old_xn = xn.copy()
        nb_iter += 1
        pn = xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
        if nb_iter < switch_iter:
            pn = proj(pn)
        else:
            pn_save = pn.copy()
            proj_pn = proj(pn).copy()
            frag = monai_predict_image(proj_pn, model, roi_size, device=device)
            pn = proj(pn_save + frag)

        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        qn = qn - sigma * proxg(qn / sigma, chan_weight, 1 / sigma)
        xn = xn + lambda_n * (pn - xn)
        results_reco.append(np.copy(xn))
        vn = vn + lambda_n * (qn - vn)
        energy = np.linalg.norm(xn - old_xn, 2)
        energy_2 = math.sqrt(np.sum(np.abs(xn - old_xn)**2))
        print("Prox iteration", nb_iter, ", norm FGP:", energy, energy_2)
    print("nb iter FB", nb_iter, "norm FB", energy)
    return xn, results_reco

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


def monai_predict_image(image, model, roi_size, sw_batch_size = 5, mode = "gaussian", overlap = 0.5, device="cpu"):
    image = torch.from_numpy(image)
    image = image.float().unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(image, roi_size, sw_batch_size, model, mode = mode, overlap = overlap)
    output = output.squeeze()
    output = torch.sigmoid(output).cpu().numpy()
    return output