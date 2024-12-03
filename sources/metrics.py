import numpy as np
from skimage.morphology import skeletonize, binary_dilation, ball, disk
from skimage.measure import label
from scipy import ndimage
from PIL import Image
from sources import image_utils
from skimage.measure import euler_number


def cldice(image, gt):
    """

        :param image: segmentation a analyser
        :param gt: verite terrain
        :return: t_sens, t_prec et la valeur du clDice
        """
    sq_image =  image_utils.normalize_image(skeletonize(image) * 1.0)
    sq_gt =  image_utils.normalize_image(skeletonize(gt) * 1.0)

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
    components_img = label(img) # permet d'identifier les différents éléments d'une source_2D
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

    # permet d'identifier les différents éléments d'une source_2D
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
