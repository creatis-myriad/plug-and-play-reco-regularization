# -*- coding: utf-8 -*-

# 22/11/18
# Odyssee Merveille

import os
import numpy as np
from skimage.morphology import reconstruction

from sources import nifti_image

"""  
    The following functions compute the RORPO operator on a 3D image.
    For more information on RORPO see [1] or [2].

    [1] "Curvilinear Structure Analysis by Ranking the Orientation Responses 
        of Path Operators", O.Merveille, H. Talbot, L. Najman, N. Passat, 
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        volume 40, 2018.
        https://ieeexplore.ieee.org/document/7862284

    [2] "RORPO: A morphological framework for curvilinear structure analysis. 
        Application to the filtering and segmentation of blood vessels" 
        Odyssée Merveille, PhD thesis, Université Paris Est, 2016
        https://hal.archives-ouvertes.fr/tel-01462887/document
"""


def compute_RPO(image, L, dilat_size=2, core=1):
    """
        Compute the Robust Path Opening from a C++ executable.
    """

    path_rorpo_exec = "/home/carneiro/opt/RORPO3D_dir/build/PO/PO"


    image_path = "./tmp_image.nii"
    output_path = "./tmp_output"

    nifti_image.save_nifti(image, image_path)
    command = path_rorpo_exec + " " + image_path + " " + output_path \
              + " " + str(L) + " " + str(dilat_size) + " --core=" + str(core)

    os.system(command)

    PO1 = nifti_image.read_nifti(output_path + "_PO1.nii")
    PO2 = nifti_image.read_nifti(output_path + "_PO2.nii")
    PO3 = nifti_image.read_nifti(output_path + "_PO3.nii")
    PO4 = nifti_image.read_nifti(output_path + "_PO4.nii")
    PO5 = nifti_image.read_nifti(output_path + "_PO5.nii")
    PO6 = nifti_image.read_nifti(output_path + "_PO6.nii")
    PO7 = nifti_image.read_nifti(output_path + "_PO7.nii")

    os.remove(image_path)
    os.remove(output_path + "_PO1.nii")
    os.remove(output_path + "_PO2.nii")
    os.remove(output_path + "_PO3.nii")
    os.remove(output_path + "_PO4.nii")
    os.remove(output_path + "_PO5.nii")
    os.remove(output_path + "_PO6.nii")
    os.remove(output_path + "_PO7.nii")

    return PO1, PO2, PO3, PO4, PO5, PO6, PO7


def cartesian_coordinates_from_num_PO_ori(num):
    '''
        return the cartesian coordinates of the vector corresponding to the
        "num-th" PO orientation.
    '''

    if (num == 0):  # horizontal
        vect = np.array([0, 0, 1], np.float)

    elif (num == 1):  # vertical
        vect = np.array([0, 1, 0], np.float)

    elif (num == 2):  #  depth
        vect = np.array([1, 0, 0], np.float)

    elif (num == 3):  # diag1
        vect = np.array([-1, 1, -1], np.float)

    elif (num == 4):  #  diag2
        vect = np.array([-1, 1, 1], np.float)

    elif (num == 5):  #  diag3
        vect = np.array([1, 1, 1], np.float)

    elif (num == 6):  #  diag4
        vect = np.array([1, 1, -1], np.float)

    return vect


def combine_vectors_after_correction(v):
    """
        Combine 2 or 3 vectors with orientation correction
        (see [2] Section 5.3.2)
    """
    nb_vect = len(v)

    # Orthogonal basis
    vect_z = np.array([1, 0, 0])
    vect_y = np.array([0, 1, 0])
    vect_x = np.array([0, 0, 1])

    if nb_vect == 2:
        # Chose the best combinaison of the 3 vectors
        # (v1 or -v1 and v2 or -v2).
        best_sum = -np.inf

        for i in [-1, 1]:
            for j in [-1, 1]:
                sum_dot_product = np.dot(i * v[0], j * v[1])

                if sum_dot_product > best_sum:
                    best_sum = sum_dot_product
                    coeff_v1 = i
                    coeff_v2 = j

        # Coord of the mean vector in the orthogonal basis.
        coord_z = np.dot(vect_z, coeff_v1 * v[0]) \
                  + np.dot(vect_z, coeff_v2 * v[1])
        coord_y = np.dot(vect_y, coeff_v1 * v[0]) \
                  + np.dot(vect_y, coeff_v2 * v[1])
        coord_x = np.dot(vect_x, coeff_v1 * v[0]) \
                  + np.dot(vect_x, coeff_v2 * v[1])

    elif nb_vect == 3:
        # Chose the best combinaison of the 3 vectors
        # (v1 or -v1, v2 or -v2 and v3 or -v3).
        best_sum = -np.inf

        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    sum_dot_product = np.dot(i * v[0], j * v[1]) \
                                      + np.dot(i * v[0], k * v[2]) \
                                      + np.dot(j * v[1], k * v[2])

                    if sum_dot_product > best_sum:
                        best_sum = sum_dot_product
                        coeff_v1 = i
                        coeff_v2 = j
                        coeff_v3 = k

        # Coord of the mean vector in the orthogonal basis.
        coord_z = np.dot(vect_z, coeff_v1 * v[0]) \
                  + np.dot(vect_z, coeff_v2 * v[1]) \
                  + np.dot(vect_z, coeff_v3 * v[2])
        coord_y = np.dot(vect_y, coeff_v1 * v[0]) \
                  + np.dot(vect_y, coeff_v2 * v[1]) \
                  + np.dot(vect_y, coeff_v3 * v[2])
        coord_x = np.dot(vect_x, coeff_v1 * v[0]) \
                  + np.dot(vect_x, coeff_v2 * v[1]) \
                  + np.dot(vect_x, coeff_v3 * v[2])
    else:
        raise NameError("Try to combine more than 3 vectors")

    return [coord_z, coord_y, coord_x]


def compute_rorpo_orientation(sorted_ori, argsort_ori, rorpo, vx_limit,
                              vy_limit, vz_limit):
    dimz, dimy, dimx = sorted_ori[0].shape  #  image dimensions

    # Compute the standard deviations of the path opening responses for the
    #  Combinations of 1, 2, 3 path opening orientations and their
    # complements (combinations of 4 and 5 PO)
    #  See Eq (53) (54) from [2]
    intra_class_std = []
    for i in range(2, 7):
        #  highest 1,2 or 3 PO responses
        high_po = sorted_ori[i:7]

        if high_po.shape[0] > 1:  #  if more than 1 PO response
            std_p = np.std(sorted_ori[i:7], axis=0) * ((7 - i) / 7.0)
        else:
            #  The std of 1 value is 0
            std_p = np.zeros_like(sorted_ori[0])
        std_op = np.std(sorted_ori[0:i], axis=0) * (i / 7.0)

        intra_class_std.append(std_p + std_op)

    #  Look for the minimum intra class std for each pixel
    #  if ind_min[0,i,j,k] == 2 we need to combine 3 orientations
    #  if ind_min[0,i,j,k] == 3 we need to combine 2 orientations
    #  if ind_min[0,i,j,k] == 4 we need to "combine" 1 orientation
    ind_min = np.argsort(intra_class_std, axis=0)

    vx = np.zeros((dimz, dimy, dimx), np.float32)
    vy = np.zeros((dimz, dimy, dimx), np.float32)
    vz = np.zeros((dimz, dimy, dimx), np.float32)
    for i in range(dimz):
        for j in range(dimy):
            for k in range(dimx):
                if rorpo[i, j, k] > 0:  #  a RORPO response exist
                    vect = [0, 0, 0]
                    if ind_min[0, i, j, k] == 4:
                        vect = cartesian_coordinates_from_num_PO_ori(
                            argsort_ori[6, i, j, k])

                    elif ind_min[0, i, j, k] == 3:
                        v1 = cartesian_coordinates_from_num_PO_ori(
                            argsort_ori[6, i, j, k])
                        v2 = cartesian_coordinates_from_num_PO_ori(
                            argsort_ori[5, i, j, k])
                        vect = combine_vectors_after_correction([v1, v2])

                    elif ind_min[0, i, j, k] == 2:
                        v1 = cartesian_coordinates_from_num_PO_ori(
                            argsort_ori[6, i, j, k])
                        v2 = cartesian_coordinates_from_num_PO_ori(
                            argsort_ori[5, i, j, k])
                        v3 = cartesian_coordinates_from_num_PO_ori(
                            argsort_ori[4, i, j, k])
                        vect = combine_vectors_after_correction([v1, v2, v3])

                    else:  #  Limit orientation
                        vect = [vz_limit[i, j, k], vy_limit[i, j, k],
                                vx_limit[i, j, k]]

                    vz[i, j, k] = vect[0]
                    vy[i, j, k] = vect[1]
                    vx[i, j, k] = vect[2]

    #  Normalize vector field
    norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    vx[norm != 0] = vx[norm != 0] / norm[norm != 0]
    vy[norm != 0] = vy[norm != 0] / norm[norm != 0]
    vz[norm != 0] = vz[norm != 0] / norm[norm != 0]

    return vx, vy, vz


def compute_RORPO(image, L, dilat_size=2, core=1):
    dimz, dimy, dimx = image.shape
    PO1, PO2, PO3, PO4, PO5, PO6, PO7 = compute_RPO(image, L, dilat_size, core)

    # Sort the orientations
    POs = np.array((PO1, PO2, PO3, PO4, PO5, PO6, PO7))
    argsort_ori = np.argsort(POs, axis=0)
    sorted_ori = np.take_along_axis(POs, \
                                    argsort_ori, axis=0)

    #  RORPO intensity feature without limit orientations
    RORPO_wo_limit_ori = sorted_ori[6] - sorted_ori[3]

    ##################### Limit orientation processing #########################
    # The following post-processing is explained in details in [2] Section 5.2.3

    limit_image = np.zeros((13, dimz, dimy, dimx))
    # ----------------- post processing 4-orientation tubes --------------------

    # horizontal + vertical + diag1 + diag4
    limit_ori_4_1 = PO1
    limit_ori_4_1 = np.minimum(limit_ori_4_1, PO2)
    limit_ori_4_1 = np.minimum(limit_ori_4_1, PO4)
    limit_ori_4_1 = np.minimum(limit_ori_4_1, PO7)
    limit_image[0] = limit_ori_4_1

    #  horizontal + vertical + diag2 + diag3
    limit_ori_4_2 = PO1
    limit_ori_4_2 = np.minimum(limit_ori_4_2, PO2)
    limit_ori_4_2 = np.minimum(limit_ori_4_2, PO5)
    limit_ori_4_2 = np.minimum(limit_ori_4_2, PO6)
    limit_image[1] = limit_ori_4_2

    # horizontal + profondeur + diag2 + diag4
    limit_ori_4_3 = PO1
    limit_ori_4_3 = np.minimum(limit_ori_4_3, PO3)
    limit_ori_4_3 = np.minimum(limit_ori_4_3, PO5)
    limit_ori_4_3 = np.minimum(limit_ori_4_3, PO7)
    limit_image[2] = limit_ori_4_3

    # horizontal + profondeur + diag1 + diag3
    limit_ori_4_4 = PO1
    limit_ori_4_4 = np.minimum(limit_ori_4_4, PO3)
    limit_ori_4_4 = np.minimum(limit_ori_4_4, PO4)
    limit_ori_4_4 = np.minimum(limit_ori_4_4, PO6)
    limit_image[3] = limit_ori_4_4

    # vertical + profondeur + diag1 + diag2
    limit_ori_4_5 = PO2
    limit_ori_4_5 = np.minimum(limit_ori_4_5, PO3)
    limit_ori_4_5 = np.minimum(limit_ori_4_5, PO4)
    limit_ori_4_5 = np.minimum(limit_ori_4_5, PO5)
    limit_image[4] = limit_ori_4_5

    # Vertical + profondeur + diag3 + diag 4
    limit_ori_4_6 = PO2
    limit_ori_4_6 = np.minimum(limit_ori_4_6, PO3)
    limit_ori_4_6 = np.minimum(limit_ori_4_6, PO6)
    limit_ori_4_6 = np.minimum(limit_ori_4_6, PO7)
    limit_image[5] = limit_ori_4_6

    # horizontal + vertical + depth + diag1
    limit_ori_4_7 = PO1
    limit_ori_4_7 = np.minimum(limit_ori_4_7, PO2)
    limit_ori_4_7 = np.minimum(limit_ori_4_7, PO3)
    limit_ori_4_7 = np.minimum(limit_ori_4_7, PO4)
    limit_image[6] = limit_ori_4_7

    # horizontal + vertical + depth + diag2
    limit_ori_4_8 = PO1
    limit_ori_4_8 = np.minimum(limit_ori_4_8, PO2)
    limit_ori_4_8 = np.minimum(limit_ori_4_8, PO3)
    limit_ori_4_8 = np.minimum(limit_ori_4_8, PO5)
    limit_image[7] = limit_ori_4_8

    # horizontal + vertical + depth + diag3
    limit_ori_4_9 = PO1
    limit_ori_4_9 = np.minimum(limit_ori_4_9, PO2)
    limit_ori_4_9 = np.minimum(limit_ori_4_9, PO3)
    limit_ori_4_9 = np.minimum(limit_ori_4_9, PO6)
    limit_image[8] = limit_ori_4_9

    # horizontal + vertical + depth + diag7
    limit_ori_4_10 = PO1
    limit_ori_4_10 = np.minimum(limit_ori_4_10, PO2)
    limit_ori_4_10 = np.minimum(limit_ori_4_10, PO3)
    limit_ori_4_10 = np.minimum(limit_ori_4_10, PO7)
    limit_image[9] = limit_ori_4_10

    min_4_tubes = np.maximum(limit_ori_4_1, limit_ori_4_2)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_3)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_4)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_5)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_6)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_7)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_8)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_9)
    min_4_tubes = np.maximum(min_4_tubes, limit_ori_4_10)

    # ------------------- post processing 5-orientation tubes ------------------
    min_5_tubes = PO4
    min_5_tubes = np.minimum(min_5_tubes, PO5)
    min_5_tubes = np.minimum(min_5_tubes, PO6)
    min_5_tubes = np.minimum(min_5_tubes, PO7)

    limit_image[10] = np.minimum(min_5_tubes, PO1)
    limit_image[11] = np.minimum(min_5_tubes, PO2)
    limit_image[12] = np.minimum(min_5_tubes, PO3)

    #  ---------------- Remove the remaining plane-like structures --------------
    geo_4 = reconstruction(sorted_ori[2], sorted_ori[3], method="dilation")
    geo_5 = reconstruction(sorted_ori[1], sorted_ori[3], method="dilation")

    rp_4 = min_4_tubes - np.minimum(min_4_tubes, geo_4)
    rp_5 = min_5_tubes - np.minimum(min_5_tubes, geo_5)

    #  ------------------- directional feature limit case -----------------------
    #  Assign the orientation of the larger response
    argmax_limit = np.argsort(limit_image, axis=0)
    vx_limit = np.zeros_like(image, np.float32)
    vy_limit = np.zeros_like(image, np.float32)
    vz_limit = np.zeros_like(image, np.float32)

    limit_case_directions = [[0, 1, -1],
                             [0, 1, 1],
                             [1, 0, -1],
                             [1, 0, 1],
                             [-1, 1, 0],
                             [1, 1, 0],
                             [-1, 1, -1],
                             [-1, 1, 1],
                             [1, 1, 1],
                             [1, 1, -1],
                             [0, 0, 1],
                             [0, 1, 0],
                             [1, 0, 0]]
    for i in range(13):
        vz_limit[argmax_limit[-1] == i] = limit_case_directions[i][0]
        vy_limit[argmax_limit[-1] == i] = limit_case_directions[i][1]
        vx_limit[argmax_limit[-1] == i] = limit_case_directions[i][2]

    #################### RORPO result with limit orientations ##################
    result = np.maximum(RORPO_wo_limit_ori, rp_4)
    result = np.maximum(result, rp_5)

    #  Compute the directiontal feature
    vx, vy, vz = compute_rorpo_orientation(sorted_ori, argsort_ori, result,
                                           vx_limit, vy_limit, vz_limit)

    ############################# Dynamic enhancement ##########################

    max_value_original = np.amax(image)
    max_value_RORPO = np.amax(result)
    result = (result / max_value_RORPO) \
             * max_value_original

    return result, vx, vy, vz


def compute_rorpo_multiscale(image, smin, factor, nb_scales, dilat_size=2,
                             core=1):
    #  Compute scales
    list_scales = [smin]
    for j in range(1, nb_scales):
        list_scales.append(int(smin * (factor ** j)))

    multiscale_rorpo = np.zeros_like(image)
    vx_multiscale = np.zeros_like(image, np.float32)
    vy_multiscale = np.zeros_like(image, np.float32)
    vz_multiscale = np.zeros_like(image, np.float32)

    for scale in list_scales:
        print("scale:", scale)
        print("dilat_size:", dilat_size)

        rorpo, vx, vy, vz = compute_RORPO(image, scale, dilat_size, core)

        #  Keep the directional feature of the smallest acceptable path lenght
        vx_multiscale[multiscale_rorpo == 0] = vx[multiscale_rorpo == 0]
        vy_multiscale[multiscale_rorpo == 0] = vy[multiscale_rorpo == 0]
        vz_multiscale[multiscale_rorpo == 0] = vz[multiscale_rorpo == 0]

        multiscale_rorpo = np.maximum(multiscale_rorpo, rorpo)

    return multiscale_rorpo, vx_multiscale, vy_multiscale, vz_multiscale