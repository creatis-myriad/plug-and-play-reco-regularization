import numpy as np
from skimage.morphology import skeletonize, binary_dilation, ball, disk
from skimage.measure import label
from scipy import ndimage
from PIL import Image
from sources import image_utils
from skimage.measure import euler_number

def read_image(path):
    '''

    :param path: chemin vers l'image a lire
    :return: numpy de l'image
    '''
    image = Image.open(path)
    return np.array(image)


def normalize_image(image):
    maxi = np.amax(image)
    mini = np.amin(image)
    image_norm = (image.astype(np.float64) - mini) / (maxi - mini)
    return image_norm


def overlap_2D(img, gt):
    '''

    :param img: segmentation 2D a analyser
    :param gt: vérité terrain 2D associée à img
    :return: la valeur de l'overlap des de img, la visualisation des TPR/FN et des TPM/FP
    '''
    # pour verifier les résultats : creation d'une map avec les differents labels :  TPR, TPM, FN, FP
    visu_TPR_FN = np.zeros((img.shape[0], img.shape[1], 3))
    visu_TPM_FP = np.zeros((img.shape[0], img.shape[1], 3))

    # skeletisation des segmentations
    img_skelet = normalize_image(skeletonize(img)*1.0)
    gt_skelet = normalize_image(skeletonize(gt)*1.0)

    # recuperation des rayons des vaisseaux
    distance_map = ndimage.distance_transform_bf(gt, 'chessboard')
    rayons = distance_map * gt_skelet
    #coordonnées des points a vérifier pour la gt (et son parcours comme on doit les verifier 1 par 1 (sauf si tu trouves mieux))
    coord_gt_sk = np.nonzero(gt_skelet)
    coord_gt_sk = np.stack(coord_gt_sk, axis=-1)

    # comptage des TPR et FN ( sur le skelette de la GT )
    TPR = 0
    FN = 0

    radius_skelet = np.zeros(gt.shape) # pour constituer la zone autour de la GT skeletisée pour récuperer apres les TPM
    for i in range(coord_gt_sk.shape[0]):
        # creation d'une image pour le point de la gt etudie
        radius_point = np.zeros(gt.shape)

        # recuperation du rayon a ce point
        rayon = rayons[coord_gt_sk[i, 0], coord_gt_sk[i, 1]]

        # definition de la zone dans laquelle le skelet de limg doit etre pour avoir un TPR
        radius_point = disc(radius_point, coord_gt_sk[i, 0], coord_gt_sk[i, 1], int(rayon))
        radius_skelet += radius_point # constitution de la zone autour de la gt pour les TPM
        # comptage des TPR et FN
        if np.sum(radius_point * img_skelet) != 0:
            TPR += 1
            visu_TPR_FN[coord_gt_sk[i, 0], coord_gt_sk[i, 1], 1] = 255
        else:
            FN += 1
            visu_TPR_FN[coord_gt_sk[i, 0], coord_gt_sk[i, 1], 0] = 255
    radius_skelet = (radius_skelet > 0) * 1.0
    # si on est dans la zone autour de la gt calcule avec radius skelet alors img_skelet et TPM
    TPM = np.sum(radius_skelet * img_skelet)
    # le reste
    FP = np.sum(img_skelet) - TPM
    # visu
    visu_TPM_FP[radius_skelet * img_skelet == 1, 1] = 255
    visu_TPM_FP[np.logical_and(radius_skelet == 0, img_skelet == 1), 0] = 255
    return (TPM + TPR) / (TPM + TPR + FP + FN), visu_TPR_FN,  visu_TPM_FP



def disc(img, x, y, pixel_radius):
    '''
    :param img: image ou on doit ajouter un disque
    :param x: coordonnee x du centre du disque a ajouter
    :param y: coordonnee y du centre du disque a ajouter
    :param pixel_radius: rayon du disque
    :return: l'image avec un disque dedans au coordonnees x, y de rayon pixel_radius
    '''
    for i in range(x - pixel_radius, x + pixel_radius + 1):
        r = int(np.sqrt(pixel_radius ** 2 - (i - x) ** 2))
        for j in range(y - r, y + r + 1):
            if not (i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1]):
                img[i, j] = 1
    return img

def bally(img, x, y, z,  pixel_radius):

    for i in range(x - pixel_radius, x + pixel_radius + 1):
        for j in range(y - pixel_radius, y + pixel_radius + 1):
            for k in range(z - pixel_radius, z + pixel_radius + 1):
                if not (i < 0 or j < 0 or k < 0 or i >= img.shape[0] or j >= img.shape[1] or k >= img.shape[2]):
                    if ((i-x)**2 + (j-y)**2 + (k-z)**2) <= (pixel_radius**2):
                        img[i, j, k] = 1
    return img

def overlap_3D(img, gt):
    '''

    :param img: segmentation 3D a analyser
    :param gt: vérité terrain 3D associée à img
    :return: la valeur de l'overlap des de img
    '''
    # skeletisation des segmentations ( pour pouvoir par la suite utiliser la fonction dans un autre fichier )
    img_skelet = normalize_image(skeletonize(img))
    gt_skelet = normalize_image(skeletonize(gt))

    # recuperation des rayons des vaisseaux
    distance_map = ndimage.distance_transform_bf(gt, 'chessboard')
    rayons = distance_map * gt_skelet
    print("distance map finie")
    #coordonnées des points a vérifier pour la gt (et son parcours comme on doit les verifier 1 par 1 (sauf si tu trouves mieux))
    coord_gt_sk = np.nonzero(gt_skelet)
    coord_gt_sk = np.stack(coord_gt_sk, axis=-1)

    # comptage des TPR et FN ( sur le skelette de la GT )
    TPR = 0
    FN = 0

    radius_skelet = np.zeros(gt.shape) # pour constituer la zone autour de la GT skeletisée pour récuperer apres les TPM
    for i in range(coord_gt_sk.shape[0]):
        # creation d'une image pour le point de la gt etudie
        radius_point = np.zeros(gt.shape)
        rayon = rayons[coord_gt_sk[i, 0], coord_gt_sk[i, 1], coord_gt_sk[i, 2]]
        radius_point = bally(radius_point, coord_gt_sk[i, 0], coord_gt_sk[i, 1], coord_gt_sk[i, 2], int(rayon))
        # recuperation du rayon a ce point
        # definition de la zone dans laquelle le skelet de limg doit etre pour avoir un TPR
        radius_skelet += radius_point # constitution de la zone autour de la gt pour les TPM
        # comptage des TPR et FN
        if np.sum(radius_point * img_skelet) != 0:
            TPR += 1
        else:
            FN += 1
    radius_skelet = (radius_skelet > 0) * 1.0
    # si on est dans la zone autour de la gt calcule avec radius skelet alors img_skelet et TPM
    TPM = np.sum(radius_skelet * img_skelet)
    # le reste
    FP = np.sum(img_skelet) - TPM

    return (TPM + TPR) / (TPM + TPR + FP + FN)

def connectivity(image, gt):
    """
    :param image: segmentation a analyser
    :param gt: verite terrain
    :return: valeur de la connectivite
    """
    nb_component_image = np.max(label(image))
    nb_component_gt = np.max(label(gt))

    con = 1 - min(1, (abs(nb_component_gt - nb_component_image)) / np.sum(gt))
    return con

def area(image, gt, alpha = 2):
    """

    :param image: segmentation a analyser
    :param gt: verite terrain
    :param alpha: taille du disque de la dilatation
    :return: la valeur de recouvrement de la segmentation sur la verite terrain
    """
    if len(image.shape) == 2:
        dilated_image = binary_dilation(image, disk(alpha))
        dilated_gt = binary_dilation(gt, disk(alpha))

        area = np.sum(np.logical_or(dilated_image * gt, dilated_gt * image)) / np.sum(np.logical_or(gt, image))
    elif len(image.shape) == 3:
        dilated_image = binary_dilation(image, ball(alpha))
        dilated_gt = binary_dilation(gt, ball(alpha))

        area = np.sum(np.logical_or(dilated_image * gt, dilated_gt * image)) / np.sum(np.logical_or(gt, image))
    else:
        area = -1
    return area

def length(image, gt, beta = 2):
    """

    :param image: segmentation a analyser
    :param gt: verite terrain
    :param beta:  taille du disque de la dilatation
    :return: taille du reseau vasculaire detecté
    """
    if len(image.shape) == 2:
        dilated_image = binary_dilation(image, disk(beta))
        dilated_gt = binary_dilation(gt, disk(beta))

        sq_image = normalize_image(skeletonize(image) * 1.0)
        sq_gt = normalize_image(skeletonize(gt) * 1.0)

        length = np.sum(np.logical_or(dilated_image * sq_gt, dilated_gt * sq_image)) / np.sum(np.logical_or(sq_gt, sq_image))
    elif len(image.shape) == 3:
        dilated_image = binary_dilation(image, ball(beta))
        dilated_gt = binary_dilation(gt, ball(beta))

        sq_image = normalize_image(skeletonize(image) * 1.0)
        sq_gt = normalize_image(skeletonize(gt) * 1.0)

        length = np.sum(np.logical_or(dilated_image * sq_gt, dilated_gt * sq_image)) / np.sum(
            np.logical_or(sq_gt, sq_image))
    else:
        length = -1
    return length

def cal(image, gt):
    """

    :param image: segmentation a analyser
    :param gt: verite terrain
    :return: mesure CAL
    """
    if length(image, gt) == -1:
        return -1
    return length(image, gt) * area(image, gt) * connectivity(image, gt)


def cldice(image, gt):
    """

        :param image: segmentation a analyser
        :param gt: verite terrain
        :return: t_sens, t_prec et la valeur du clDice
        """
    sq_image = normalize_image(skeletonize(image) * 1.0)
    sq_gt = normalize_image(skeletonize(gt) * 1.0)

    t_sens = np.sum(sq_gt * image) / np.sum(sq_gt)
    t_prec = np.sum(sq_image * gt) / np.sum(sq_image)

    cl_dice = 2 * ( t_prec * t_sens) / (t_prec + t_sens)
    return t_sens, t_prec, cl_dice

def extract_skelet(seg, gt):
    gt_sk = image_utils.normalize_image(skeletonize(gt))
    seg_centerline = gt_sk * seg
    return seg_centerline


def get_stat(gt_values, values, n):
    voxels_values = sum(values[-n:])
    voxels_gt = sum(gt_values)
    return voxels_values/voxels_gt

def val_pixels_per_component(img):
    components_img = label(img) # permet d'identifier les différents éléments d'une image
    pixels_per_component = []
    for i in range(1, np.max(components_img) + 1):# comptage des nombres de pixels pour chaque element
        nombre_pix = np.sum(components_img == i)
        pixels_per_component.append(nombre_pix)
    return pixels_per_component


def get_componants_stats(gt_values, values):
    gt_values = sorted(gt_values)
    values = sorted(values)
    n_comp = 1
    value_1_stat = get_stat(gt_values, values, n_comp)
    n_comp = 3
    value_3_stat = get_stat(gt_values, values, n_comp)
    n_comp = 5
    value_5_stat = get_stat(gt_values, values, n_comp)
    return value_1_stat, value_3_stat, value_5_stat

def extract_mains_component_rank(sk):

    # permet d'identifier les différents éléments d'une image
    components_inter = label(sk)
    pixels_per_component = []

    # comptage des nombres de pixels pour chaque element (hors background)
    for i in range(1, np.max(components_inter) + 1):
        nombre_pix = np.sum(components_inter == i)
        pixels_per_component.append(nombre_pix)
    pixels_per_component = np.array(pixels_per_component)

    #tri des arguments dans l'ordre croissant
    pixels_per_component = np.sort(pixels_per_component)


    return pixels_per_component, np.max(components_inter)

def calculate_rmcc(seg, gt):

    # intersection verite terrain et gt
    # calcul des squelettes avec un petit pretraitement des familles
    inter_sk = extract_skelet(seg, gt)
    gt_sk = image_utils.normalize_image(skeletonize(gt))

    # Extraire les composantes principales de la gt
    components_inter = np.max(label(gt_sk))
    repartition_inter_sk, nb_element_inter_sk = extract_mains_component_rank(inter_sk)

    # Calcul du rmcc
    rmcc = sum(repartition_inter_sk[-components_inter:]) / np.sum(gt_sk)
    return rmcc

def calculate_nb_composant_pourcent(seg, gt):

    inter_sk = extract_skelet(seg, gt)
    gt_sk_1 = image_utils.normalize_image(skeletonize(gt))

    components_inter = label(inter_sk)
    nombre_composante = np.max(components_inter)
    pourcent_recouvrement = np.sum(inter_sk) / np.sum(gt_sk_1)
    return nombre_composante, pourcent_recouvrement

def euler_number_error_numpy(y_true, y_pred):
    euler_number_true = euler_number(y_true)
    euler_number_pred = euler_number(y_pred)

    euler_number_error = np.absolute(np.absolute(euler_number_true - euler_number_pred) / euler_number_true)

    return euler_number_error, euler_number_true, euler_number_pred


def b0_error_numpy(y_true, y_pred):
    _, ncc_true = label(y_true, return_num=True)
    _, ncc_pred = label(y_pred, return_num=True)

    b0_true= ncc_true
    b0_pred = ncc_pred

    b0_error = np.absolute(b0_true - b0_pred) / b0_true

    return b0_error, b0_true, b0_pred

def b1_error_numpy(y_true, y_pred):

    __, euler_number_true, euler_number_pred= euler_number_error_numpy(y_true, y_pred)
    # euler_number_pred = euler_number_error_numpy(y_pred)

    _, ncc_true = label(y_true, return_num=True)
    _, ncc_pred = label(y_pred, return_num=True)

    b0_true= ncc_true - 1
    b0_pred = ncc_pred - 1

    y_true_inverse = np.ones(y_true.shape) - y_true
    y_pred_inverse = np.ones(y_pred.shape) - y_pred

    _, ncc_true = label(y_true_inverse, return_num=True)
    _, ncc_pred = label(y_pred_inverse, return_num=True)

    b2_true= ncc_true - 1
    b2_pred = ncc_pred - 1

    b1_true = b0_true + b2_true - euler_number_true
    b1_pred = b0_pred + b2_pred - euler_number_pred
    # print(f"b1_true:{b1_true}")
    # print(f"b1_pred:{b1_pred}")


    b1_error = np.absolute(b1_true - b1_pred) / b1_true

    return b1_error, b1_true, b1_pred

def b2_error_numpy(y_true, y_pred):
    y_true_inverse = np.ones(y_true.shape) - y_true
    y_pred_inverse = np.ones(y_pred.shape) - y_pred

    _, ncc_true = label(y_true_inverse, return_num=True)
    _, ncc_pred = label(y_pred_inverse, return_num=True)

    b2_true= ncc_true - 1
    b2_pred = ncc_pred - 1
    b2_error = np.absolute(b2_true - b2_pred) / b2_true

    return b2_error, b2_true, b2_pred

def b0_error_numpy_2D(y_true, y_pred):
    _, ncc_true = label(y_true, return_num=True,  connectivity=2)
    _, ncc_pred = label(y_pred, return_num=True, connectivity=2)

    b0_true= ncc_true
    b0_pred = ncc_pred

    b0_error = np.absolute(b0_true - b0_pred) / b0_true

    return b0_error, b0_true, b0_pred

def b1_error_numpy_2D(y_true, y_pred):
    y_true_inverse = np.ones(y_true.shape) - y_true
    y_pred_inverse = np.ones(y_pred.shape) - y_pred

    _, ncc_true = label(y_true_inverse, return_num=True, connectivity=1)
    _, ncc_pred = label(y_pred_inverse, return_num=True, connectivity=1)

    b1_true = ncc_true - 1
    b1_pred = ncc_pred - 1
    b1_error = np.absolute(b1_true - b1_pred) / b1_true

    return b1_error, b1_true, b1_pred
