import numpy as np
import torch
from monai.inferers import sliding_window_inference
import pandas as pd
from time import time


def primal_dual_ind_chan_tv_3D(image, c1, c2, L, chan_weight, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100):
    following = pd.DataFrame(columns=['iteration', 'energy'])
    xn = np.zeros(image.shape, np.float)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float)
    energy = 100
    nb_iter = 0
    grad_h = ((c1 - image) ** 2 - (c2 - image) ** 2)
    t = 0
    while (energy > epsilon and nb_iter < max_iter):
        old_xn = xn.copy()
        nb_iter += 1
        pn =xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
        pn = proj(pn)
        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        t1 = time()
        qn = qn - sigma*proxg(qn/sigma, chan_weight, 1/sigma)
        t+= time() - t1
        print(time() - t1)
        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)
        energy = np.linalg.norm(xn.reshape(-1) - old_xn.reshape(-1), 2)

        values_to_add = {'iteration': nb_iter, 'energy':energy}
        row_to_add = pd.Series(values_to_add)
        following = following.append(row_to_add, ignore_index=True)
        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    print(t)

    return xn, following


def primal_dual_dir_tv_3D(image, c1, c2, L, chan_weight, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100):
    following = pd.DataFrame(columns=['iteration', 'energy'])
    xn = np.zeros(image.shape, np.float)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float)
    energy = 100
    nb_iter = 0
    grad_h = ((c1 - image) ** 2 - (c2 - image) ** 2)
    t = 0
    while (energy > epsilon and nb_iter < max_iter):
        old_xn = xn.copy()
        nb_iter += 1
        pn =xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
        pn = proj(pn)
        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        t1 = time()
        qn = qn - sigma*proxg_dir(qn/sigma, chan_weight, 1/sigma)
        t+= time() - t1
        print(time() - t1)
        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)
        energy = np.linalg.norm(xn.reshape(-1) - old_xn.reshape(-1), 2)

        values_to_add = {'iteration': nb_iter, 'energy':energy}
        row_to_add = pd.Series(values_to_add)
        following = following.append(row_to_add, ignore_index=True)
        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    print(t)

    return xn, following

def primal_dual_ind_reconnect_3D_no_frag(image, gt, mask, c1, c2, L, chan_weight, data_weight, model, roi_size, switch_iter = 300, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100, device="cpu"):
    following = pd.DataFrame(columns=['iteration', 'energy'])
    # following_metric = pd.DataFrame(columns=['iteration', 'dice'])
    xn = np.zeros(image.shape, np.float)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float)
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
            pn = xn - tau * (data_weight * grad_h + (L_t.dot(vn)).reshape(image.shape))
            reconnections = monai_predict_image(proj(pn), model, roi_size, device=device)
            pn = reconnections.copy()
            pn = proj(pn)


        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        qn = qn - sigma*proxg(qn/sigma, chan_weight, 1/sigma)
        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)

        energy = np.linalg.norm(xn.reshape(-1) - old_xn.reshape(-1), 2)
        values_to_add = {'iteration': nb_iter, 'energy':energy}
        row_to_add = pd.Series(values_to_add)
        following = following.append(row_to_add, ignore_index=True)
        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    return xn, following, iterations

def primal_dual_ind_reconnect_3D(image, gt, mask, c1, c2, L, chan_weight, data_weight, model, roi_size, switch_iter = 300, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100, device="cpu"):
    following = pd.DataFrame(columns=['iteration', 'energy'])
    following_metric = pd.DataFrame(columns=['iteration', 'dice'])
    xn = np.zeros(image.shape, np.float64)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float64)
    energy = 100
    nb_iter = 0
    grad_h = ((c1 - image) ** 2 - (c2 - image) ** 2)
    while ((energy > epsilon or nb_iter < switch_iter) and nb_iter < max_iter):
        old_xn = xn.copy()
        nb_iter += 1
        if nb_iter < switch_iter:
            pn = xn - tau * (grad_h + (L_t.dot(vn)).reshape(image.shape))
            pn = proj(pn)
        else:
            pn = xn - tau * (data_weight * grad_h + (L_t.dot(vn)).reshape(image.shape))
            pn_save = pn.copy()
            reconnections = monai_predict_image(proj(pn), model, roi_size, device=device) # le reseau retourne seulement les reconnexions
            pn = pn_save + reconnections
            pn = proj(pn)
        qn = vn + sigma * L.dot((2 * pn - xn).flatten())
        qn = qn - sigma*proxg(qn/sigma, chan_weight, 1/sigma)
        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)

        energy = np.linalg.norm(xn.reshape(-1) - old_xn.reshape(-1), 2)

        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    return xn, following, following_metric


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

def proxg_dir(u, chan_weight, gamma):
    vx = u[:int(u.shape[0] / 4)]
    vy = u[int(u.shape[0] / 4):int(2*(u.shape[0] / 4))]
    vz = u[int(2 * (u.shape[0] / 4)):int(3 * (u.shape[0] / 4))]
    v_dir = u[int(3 * (u.shape[0] / 4)):]

    norm = np.sqrt(vx * vx + vy * vy + vz * vz + v_dir * v_dir)
    norm = np.tile(norm, 4)
    prox_norm = (1 - (gamma*chan_weight / np.maximum(norm, gamma*chan_weight)))*u
    return prox_norm

def monai_predict_image(image, model, roi_size, sw_batch_size = 5, mode = "gaussian", overlap = 0.5, device = "cpu"):

    new_image = np.zeros(image.shape + np.array([10, 10, 10]))
    new_image[
    new_image.shape[0] // 2 - image.shape[0] // 2: new_image.shape[0] // 2 + image.shape[0] // 2 + image.shape[
        0] % 2,
    new_image.shape[1] // 2 - image.shape[1] // 2: new_image.shape[1] // 2 + image.shape[1] // 2 + image.shape[
        1] % 2,
    new_image.shape[2] // 2 - image.shape[2] // 2: new_image.shape[2] // 2 + image.shape[2] // 2 + image.shape[
        2] % 2] = image.copy()

    new_image = torch.from_numpy(new_image)
    new_image = new_image.float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(inputs=new_image, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, mode = mode, overlap = overlap)
    output = output.squeeze()
    output = torch.sigmoid(output).cpu().numpy()
    output = output[
    output.shape[0] // 2 - image.shape[0] // 2: output.shape[0] // 2 + image.shape[0] // 2 + image.shape[0] % 2,
    output.shape[1] // 2 - image.shape[1] // 2: output.shape[1] // 2 + image.shape[1] // 2 + image.shape[1] % 2,
    output.shape[2] // 2 - image.shape[2] // 2: output.shape[2] // 2 + image.shape[2] // 2 + image.shape[2] % 2]
    return output