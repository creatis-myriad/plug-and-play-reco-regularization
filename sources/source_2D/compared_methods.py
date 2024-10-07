from sources.source_2D.plug_and_play import proj, proxg
import sources.source_2D.grad_div_interpolation_2d as grd2D
from sources import image_utils
from skimage.filters import threshold_otsu

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


def directional_segmentation(image_path, rorpo_path, tv_weight, sigma=10e-3, max_iter=1000, lambda_n=1):
    tau = 2/(1.1+16 * sigma)

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

    # parameters
    c1 = otsu
    c2 = 0

    op_gradx = grd2D.gradient_2d_along_axis_anisotropy_correction(ori, axis=1)
    op_grady = grd2D.gradient_2d_along_axis_anisotropy_correction(ori, axis=0)

    # Directional gradient
    op_grad_dir = grd2D.directional_2d_gradient(ori)

    # Mixed gradient
    L = grd2D.mixed_gradient_operator_2d(op_grady, op_gradx, op_grad_dir)

    xn = primal_dual_ind_chan_tv(image_norm, c1, c2, L, tv_weight, tau, sigma, 1.e-4, lambda_n, max_iter)
    segment_8_bits_dir = ((xn >= 0.5) * 255).astype(np.uint8)

    return segment_8_bits_dir

def chan_segmentation(image_path, tv_weight, sigma=10e-3, max_iter=1000, lambda_n=1):
    tau = 2 / (1.1 + 16 * sigma)

    image = image_utils.read_image(image_path)
    image_norm = image_utils.normalize_image(image)
    otsu = threshold_otsu(image_norm)
    dimy, dimx = image.shape

    # parameters
    c1 = otsu
    c2 = 0

    ## gradient
    op_grady = grd2D.gradient_2d_along_axis([dimy, dimx], axis=0)
    op_gradx = grd2D.gradient_2d_along_axis([dimy, dimx], axis=1)

    L = grd2D.standard_gradient_operator_2d(op_grady, op_gradx)
    xn = primal_dual_ind_chan_tv(image_norm, c1, c2, L, tv_weight, tau, sigma, 1.e-4, lambda_n, max_iter)

    segment_8_bits_chan = ((xn >= 0.5) * 255).astype(np.uint8)

    return segment_8_bits_chan



import imageUtils
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from math import sqrt
from scipy.optimize import curve_fit


def polynomial(x, a, b, c):
    return a * x**2 + b * x + c


def test_coord(point_depart, coord_L1):
    return any(np.array_equal(point_depart, coord) for coord in coord_L1)


def calculate_connected_composant(image):
    image_labeled = label(image, connectivity=2)
    image_labeled = image_labeled.astype(np.uint8)
    i_max = 0
    nb_pix_max = 0
    for i in range(1, np.max(image_labeled) + 1):
        nb_pix = np.sum((image_labeled == i) * 1.0)
        if nb_pix > nb_pix_max:
            nb_pix_max = nb_pix
            i_max = i
    return image_labeled, i_max


def calculate_minima_distance(image, composante_L1, composante_Li):
    #calculate the skelet to work on it
    skelet = skeletonize(image)
    squelette_labeled_L1 = composante_L1 * skelet
    squelette_labeled_Li = composante_Li * skelet

    coord_L1 = np.nonzero(squelette_labeled_L1)
    coord_L1 = np.stack(coord_L1, axis=-1)
    coord_Li = np.nonzero(squelette_labeled_Li)
    coord_Li = np.stack(coord_Li, axis=-1)
    dist_min = 10000000
    for (xi, yi) in coord_Li:
        for (x1, y1) in coord_L1:
            dist = sqrt((xi - x1) ** 2 + (yi - y1) ** 2)
            if dist < dist_min:
                dist_min = dist
                coord_A = np.array([x1, y1])
                coord_B = np.array([xi, yi])
    return coord_A, coord_B


def courbe_poly(image, x, y):
    courbe = np.zeros(image.shape)
    if len(x) >= 3:
        # Ajustate the polynomial curve
        popt, _ = curve_fit(polynomial, x, y)
        ymax, xmax = image.shape

        # generates the points to draw the curve
        x_fit = np.linspace(0, xmax, 1000)
        y_fit = polynomial(x_fit, *popt)

        selection_y = np.logical_and((y_fit < ymax), (y_fit >= 0))
        selection_x = np.logical_and((x_fit < xmax), (x_fit >= 0))
        selection = selection_x * selection_y

        coord_curve = np.dstack((y_fit, x_fit)).astype("int32")[0]
        coord_curve = coord_curve[selection]
        # definition of the discrete curve
        for el in range(coord_curve.shape[0]):
            courbe[coord_curve[el][0], coord_curve[el][1]] = 1
    return courbe




def prw(segmentation_path, alpha,roi_size, error):
    """
    :param segmentation_path: path toward the probability map of the segmentation that we want to post processed
    :param alpha: weight between the probability map and the direction to guide the walker
    :param roi_size: roi size for possible reconnection
    :param error:
    """
    probability_map = image_utils.read_image(segmentation_path)
    segmentation = ((probability_map >= 0.5) * 255).astype(np.uint8)
    segmentation_post_processed = imageUtils.normalizeImage(segmentation.copy(), 1)

    neighbors = np.ones([3, 3])
    neighbors[1, 1] = 0

    # label all connected component
    labeled_image, i_max = calculate_connected_composant(segmentation)
    segmentation = imageUtils.normalizeImage(segmentation, 1)
    skelet = skeletonize(segmentation)

    # research of the minimal distance for each point of the skelet

    labeled_skelet = labeled_image * skelet

    # coordinates of the principal composant
    coord_L1 = np.nonzero(labeled_skelet == i_max)
    coord_L1 = np.stack(coord_L1, axis=-1)
    for i in range(1, np.max(labeled_image) + 1):
        print(f" composante {i}/{np.max(labeled_image) + 1}")
        if i != i_max:
            image_labeled_bis, i_max = calculate_connected_composant(segmentation_post_processed)
            l1 = (image_labeled_bis == i_max)
            li = (labeled_image == i)

            coord_A, coord_B = calculate_minima_distance(segmentation, l1, li)

            # Calculate the polynomial equation to have the walker direction
            coord_M = coord_A + (coord_B - coord_A)//2
            if coord_A[0] in range(coord_M[0] - roi_size[0]//2,  coord_M[0] + roi_size[0]//2) and coord_B[0] in range(coord_M[0] - roi_size[0]//2,  coord_M[0] + roi_size[0]//2)  and coord_A[1] in range(coord_M[1] - roi_size[1]//2,  coord_M[1] + roi_size[1]//2) and coord_B[1] in range(coord_M[1] - roi_size[1]//2,  coord_M[1] + roi_size[1]//2):
                coord_Li = np.nonzero(labeled_skelet == i)
                x = coord_Li[1]
                y = coord_Li[0]
                if len(x)>=3:
                    courbe = courbe_poly(segmentation, x, y)
                    # research of point C
                    points_possibles = (image_labeled_bis == i_max) * courbe
                    coords_c_possibles = np.nonzero(points_possibles)
                    coords_c_possibles = np.stack(coords_c_possibles, axis=-1)
                    dist_min = 1000000
                    for (xi, yi) in coords_c_possibles:
                        dist = sqrt((xi - coord_B[0]) ** 2 + (yi - coord_B[1]) ** 2)
                        if dist < dist_min:
                            dist_min = dist
                            coord_C = np.array([xi, yi])
                    if dist_min!= 1000000:
                        print(f"composant to do !!! ")
                        # calculate the probabilité map
                        pd = np.zeros(probability_map.shape)
                        for x in range(pd.shape[0]):
                            for y in range(pd.shape[1]):
                                if x in range(coord_M[0] - roi_size[0] // 2, coord_M[0] + roi_size[0] // 2) and x in range(coord_M[0] - roi_size[0] // 2, coord_M[0] + roi_size[
                                    0] // 2) and y in range(coord_M[1] - roi_size[1] // 2, coord_M[1] + roi_size[1] // 2) and y in range(
                                        coord_M[1] - roi_size[1] // 2, coord_M[1] + roi_size[1] // 2):
                                    if not (x == coord_C[0] and y == coord_C[1]):
                                        proba = 1 / sqrt((x - coord_C[0]) ** 2 + (y - coord_C[1]) ** 2)
                                        pd[x, y] = proba
                        pd[coord_C[0], coord_C[1]] = 1
                        walker_map = pd * alpha + (1 - alpha) * probability_map

                        # calculate the seeds for the i-th composant
                        ROI_matrix = np.zeros(segmentation.shape)
                        ROI_matrix[int(coord_M[0] - roi_size[0] // 2): int(coord_M[0] + roi_size[0] // 2),
                        int(coord_M[1]- roi_size[1] // 2): int(coord_M[1]+ roi_size[1] // 2)] = 1
                        seeds = li * ROI_matrix

                        seeds_coord = np.nonzero(seeds)
                        seeds_coord = np.stack(seeds_coord, axis=-1)

                        chemin_tot = np.zeros(segmentation.shape)

                        # treatement for each seed
                        for seed_i in range(seeds_coord.shape[0]):
                            chemin = np.zeros(segmentation.shape)
                            point_depart = seeds_coord[seed_i]
                            test = True

                            while (not test_coord(point_depart, coord_L1)) and test:
                                # go toward the neighbour which has the best neighbourhood
                                neighbors_walker = np.zeros(segmentation.shape)
                                neighbors_walker[point_depart[0] - 1: point_depart[0] + 2,
                                point_depart[1] - 1: point_depart[1] + 2] = 1
                                neighbors_walker[point_depart[0] - 1, point_depart[1]] = 0
                                neighbor_etape_i = neighbors_walker * walker_map * (np.ones(
                                    segmentation.shape) - chemin)
                                neigbor_etape_i_pnn = probability_map * neighbors_walker
                                coord = np.unravel_index(neighbor_etape_i.argmax(), neighbor_etape_i.shape)
                                test = not (np.all(neigbor_etape_i_pnn < error) or (coord[0]< coord_M[0]- roi_size[0] // 2 ) or (coord[0]> coord_M[0] + roi_size[0] // 2) or (coord[1]< coord_M[1]- roi_size[1] // 2 ) or (coord[1]> coord_M[1] + roi_size[1] // 2))
                                if test:
                                    chemin[coord[0], coord[1]] = 1
                                    point_depart = coord
                            if test:
                                chemin_tot += chemin.copy()
                            segmentation_post_processed = ((segmentation_post_processed + chemin_tot) > 0.5) * 1.0
    return segmentation_post_processed



