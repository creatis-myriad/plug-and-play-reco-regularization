def proxg_dir(u, chan_weight, gamma):
    vx = u[:int(u.shape[0] / 4)]
    vy = u[int(u.shape[0] / 4):int(2*(u.shape[0] / 4))]
    vz = u[int(2 * (u.shape[0] / 4)):int(3 * (u.shape[0] / 4))]
    v_dir = u[int(3 * (u.shape[0] / 4)):]

    norm = np.sqrt(vx * vx + vy * vy + vz * vz + v_dir * v_dir)
    norm = np.tile(norm, 4)
    prox_norm = (1 - (gamma*chan_weight / np.maximum(norm, gamma*chan_weight)))*u
    return prox_norm


def primal_dual_ind_chan_tv_3D(image, c1, c2, L, chan_weight, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100):
    xn = np.zeros(image.shape, np.float64)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float64)
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
        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)
    print(t)

    return xn, xn


def primal_dual_dir_tv_3D(image, c1, c2, L, chan_weight, tau = 0.25, sigma = 0.25, epsilon=1.e-1,lambda_n = 0.5,
                                max_iter=100):
    xn = np.zeros(image.shape, np.float64)
    L_t = L.getH()
    vn = np.zeros(L.shape[0], np.float64)
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
        qn = qn - sigma*proxg_dir(qn/sigma, chan_weight, 1/sigma)

        xn = xn + lambda_n * (pn - xn)
        vn = vn + lambda_n * (qn - vn)
        energy = np.linalg.norm(xn.reshape(-1) - old_xn.reshape(-1), 2)

        print("Prox iteration", nb_iter, ", norm FGP:", energy)
    print("nb iter FB", nb_iter, "norm FB", energy)

    return xn, xn