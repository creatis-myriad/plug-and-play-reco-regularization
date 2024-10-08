"""
Copyright or © or Copr. Odyssee Merveille (2019)
odyssee.merveille@gmail.com

This software is a computer program whose purpose is to reproduce the results 
of the article "nD variational restoration of curvilinear structures with 
prior-based directional regularization", O. Merveille, B. Naegel, H. Talbot 
and N. Passat, IEEE Transactions on Image Processing, 2019
https://hal.archives-ouvertes.fr/hal-01832636.

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

import math 
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pylab as plt
import os
from glob import glob

def read_image(path):
    image = Image.open(path)
    return np.array(image)


def save_image(image, output_path):
	image_pil = Image.fromarray(image)
	image_pil.save(output_path)

def show_image(im, title="", colormap = "gray", save= "", dpi = 100):
	
	fig = plt.figure()
	fig.patch.set_facecolor('white')
		
	plt.imshow(im, colormap)
	plt.title(title)
	plt.tight_layout()
	plt.show()
	
	if save != "":
		plt.savefig(save, dpi = dpi)
	return	


def normalize_image(image, maxi=None):
    if maxi == None:
        maxi = np.amax(image)
    mini = np.amin(image)
    maxi_image =  np.amax(image)
    image_norm = ((image.astype(np.float64) - mini) / (maxi_image - mini))* maxi
    return image_norm



def read_nifti_image(path):
    image = nib.load(path)
    image_numpy = np.array(image.dataobj)
    return image_numpy


def save_nifti(array, path, metadata_model=None):
    """
    Save a nd array to a nifti image on the disk
    Parameters
    ----------
    array : nd array
        the image to save
    path : string
        location to save the image
    metadata_model : string, optional
        the path to a nifti image.
        The image will be saved with the same header as metadata_model. Otherwise, a default header will be provided

    Returns
    -------
    None.

    """
    if metadata_model is None:
        res = nib.Nifti1Image(array, affine=np.eye(4))
    else:
        nib_image = nib.load(metadata_model)
        res = nib.Nifti1Image(array, affine=nib_image.affine, header=nib_image.header)

    nib.save(res, path)


def IoU(image, gt):

    intersection = np.logical_and(image != 0, gt != 0)
    union = np.logical_or(image != 0, gt != 0)

    tp = np.count_nonzero(intersection)
    union_measure = np.count_nonzero(union)
    iou_measure = tp / union_measure

    return iou_measure, tp, union


# def compute_dice(image1, image2, mask= np.array(())):
#     '''
#         Compute the dice from two binary images
#
#         INPUT:
#             - image1: nd numpy array; the binary source_2D to compare
#             - image2: nd numpy array; the binary ground truth
#     '''
#     if mask == np.array(()):
#         tp = np.count_nonzero(np.logical_and(image1 != 0, image2 != 0))
#         fp = np.count_nonzero(np.logical_and(image1 != 0, image2 == 0))
#         tn = np.count_nonzero(np.logical_and(image1 == 0, image2 == 0))
#         fn = np.count_nonzero(np.logical_and(image1 == 0, image2 != 0))
#
#     else:
#         tp = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 != 0, image2 != 0),
#                                 mask != 0))
#         fp = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 != 0, image2 == 0),
#                                 mask != 0))
#         tn = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 == 0, image2 == 0),
#                                 mask != 0))
#         fn = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 == 0, image2 != 0),
#                                 mask != 0))
#
#
#     dice = 2 * tp / (2 * tp + fp + fn)
#
#
#
#     return dice
#
# def compute_accuracy(image1, image2, mask= np.array(())):
#     '''
#         Compute the Accuracy from two binary images
#
#         INPUT:
#             - image1: nd numpy array; the binary source_2D to compare
#             - image2: nd numpy array; the binary ground truth
#     '''
#     if mask == np.array(()):
#         tp = np.count_nonzero(np.logical_and(image1 != 0, image2 != 0))
#         fp = np.count_nonzero(np.logical_and(image1 != 0, image2 == 0))
#         tn = np.count_nonzero(np.logical_and(image1 == 0, image2 == 0))
#         fn = np.count_nonzero(np.logical_and(image1 == 0, image2 != 0))
#
#     else:
#         tp = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 != 0, image2 != 0),
#                                 mask != 0))
#         fp = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 != 0, image2 == 0),
#                                 mask != 0))
#         tn = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 == 0, image2 == 0),
#                                 mask != 0))
#         fn = np.count_nonzero(np.logical_and(
#                                 np.logical_and(image1 == 0, image2 != 0),
#                                 mask != 0))
#
#
#     acc = (tp + tn) / (tp + tn + fp + fn)
#
#     tpr = tp / (tp + fn)
#     tnr = tn / (fp + tn)
#
#     return acc, tpr, tnr
#
#
# def compute_mcc(image1, image2, mask= np.array(())) :
#     '''
#     Compute the Matthews correlation coefficient [1] from two images
#
#     INPUT:
#         - image1: nd numpy array; the binary source_2D to compare
#         - image2: nd numpy array; the binary ground truth
#
#     OUTPUT:
#         - mcc: int; the MCC
#
#     [1] B.W. Matthews, "Comparison of the predicted and observed secondary
#     structure of T4 phage lysozyme", Biochimica et Biophysica Acta, 1975
#     https://www.sciencedirect.com/science/article/pii/0005279575901099?via%3Dihub
#
#     '''
#     if mask == np.array(()):
#         tp = np.count_nonzero(np.logical_and(image1 != 0, image2 != 0))
#         fp = np.count_nonzero(np.logical_and(image1 != 0, image2 == 0))
#         tn = np.count_nonzero(np.logical_and(image1 == 0, image2 == 0))
#         fn = np.count_nonzero(np.logical_and(image1 == 0, image2 != 0))
#     else:
#         tp = np.count_nonzero(np.logical_and(
#             np.logical_and(image1 != 0, image2 != 0),
#             mask != 0))
#         fp = np.count_nonzero(np.logical_and(
#             np.logical_and(image1 != 0, image2 == 0),
#             mask != 0))
#         tn = np.count_nonzero(np.logical_and(
#             np.logical_and(image1 == 0, image2 == 0),
#             mask != 0))
#         fn = np.count_nonzero(np.logical_and(
#             np.logical_and(image1 == 0, image2 != 0),
#             mask != 0))
#
#     print(tp, tn, fp, fn)
#     mcc_numerator = tp * tn - fp * fn
#     mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#
#     if mcc_denominator != 0:
#         mcc = mcc_numerator / mcc_denominator
#     else:
#         mcc = 0
#
#     return  mcc
#
#
# def compute_best_mcc(image1, image2, mask= np.array(())):
#     '''
#     Return the best MCC over all source_2D thresholds
#
#     INPUT:
#         - image1: nd numpy array; the uint8 source_2D to compare
#         - image2: nd numpy array; the binary ground truth
#
#     OUTPUT:
#         - best_mcc: int; the best MCC
#         - best_seuil: int; the threshold associated to best_mcc
#     '''
#
#     nb_seuils = np.amax(image1)
#     best_mcc = 0
#     best_seuil = -1
#     for s in range(nb_seuils):
#         image_seuil = (image1 > s).astype(np.uint8)
#
#         mcc = compute_mcc(image_seuil, image2, mask)
#
#         if abs(mcc) > abs(best_mcc):
#             best_mcc = mcc
#             best_seuil =s
#
#     return best_mcc, best_seuil
#
#
# def roc_curve(image1, image2):
#     """
#         Compute the x and y values of the ROC curve
#
#         INPUT:
#             - image1: nd numpy array; the uint8 source_2D to compare
#             - image2: nd numpy array; the binary ground truth
#
#         OUTPUT:
#             - fp_liste: 1d numpy array; the false positives values for each
#                         source_2D 1 threshold
#             - tp_liste: 1d numpy array; the true positives values for each
#                         source_2D 1 threshold
#     """
#
#     tp_max = np.count_nonzero(image2)
#     fp_liste = []
#     tp_liste = []
#
#     for s in range(255):
#         image_seuil = (image1 > s).astype(np.uint8)
#         tp = np.count_nonzero(np.logical_and(image_seuil != 0, image2 != 0))
#         fp = np.count_nonzero(np.logical_and(image_seuil != 0, image2 == 0))
#
#         fp_liste.append(fp / tp_max)
#         tp_liste.append(tp / tp_max)
#
#     return fp_liste, tp_liste
