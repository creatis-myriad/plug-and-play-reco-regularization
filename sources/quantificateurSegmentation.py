from skimage.metrics import structural_similarity as ssim1
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import math

########################################################################################################################
#tout doit etre en numpy magueule

def MCC(img_bin, gt):
    mcc = matthews_corrcoef(img_bin.ravel(), gt.ravel())
    return mcc

def confusionMatrix(img_bin, gt):
    tn, fp, fn, tp = confusion_matrix(gt.ravel(), img_bin.ravel()).ravel()
    return (tn, fp, fn, tp)


def dice(img_bin, gt):
    tn, fp, fn, tp = confusionMatrix(img_bin, gt)
    return 2 * tp / (2 * tp + fn + fp)


def hausdorff(img_bin, gt):
    return max(directed_hausdorff(img_bin, gt)[0], directed_hausdorff(gt, img_bin)[0])


def roc(img, gt):
    seuillages = np.arange(0,1,0.01)
    MCC = 0
    best_seuill = np.zeros(img.shape)
    for i in seuillages:
        img_seuil = (img > float(i)) * 1.0
        mcc = MCC(img_seuil, gt)
        if mcc>= MCC:
            best_seuill = img_seuil
            MCC = mcc
    return MCC, best_seuill

def confusionMatrixMasked(img_bin, gt, mask = np.array(())):
    if mask == np.array(()):
        tp = np.count_nonzero(np.logical_and(img_bin != 0, img_bin != 0))
        fp = np.count_nonzero(np.logical_and(img_bin != 0, img_bin == 0))
        tn = np.count_nonzero(np.logical_and(img_bin == 0, img_bin == 0))
        fn = np.count_nonzero(np.logical_and(img_bin == 0, img_bin != 0))

    else:
        tp = np.count_nonzero(np.logical_and(
            np.logical_and(img_bin != 0, gt != 0),
            mask != 0))
        fp = np.count_nonzero(np.logical_and(
            np.logical_and(img_bin != 0, gt == 0),
            mask != 0))
        tn = np.count_nonzero(np.logical_and(
            np.logical_and(img_bin == 0, gt == 0),
            mask != 0))
        fn = np.count_nonzero(np.logical_and(
            np.logical_and(img_bin == 0, gt != 0),
            mask != 0))

    return (tn, fp, fn, tp)

def diceMasked(img_bin, gt, mask):
    tn, fp, fn, tp = confusionMatrixMasked(img_bin, gt, mask)
    return 2 * tp / (2 * tp + fn + fp)

def evaluate_image_binaire3D(img_bin, gt, mask):
    (tn, fp, fn, tp) = confusionMatrixMasked(img_bin, gt, mask)
    if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) == 0 :
        mcc = -1
    else:
        mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)))
    d = diceMasked(img_bin, gt, mask)
    acc = (tp + tn)/(tn + fp + fn + tp)
    sp = tn / (tn + fp)
    se = tp / (tp + fn)
    fpr = fp / (fp + tn)
    metrics = {'matrix_confusion': (tn, fp, fn, tp),'acc': acc,'sp': sp, 'se' : se, 'mcc' : mcc, 'dice' : d, 'fpr': fpr }
    return metrics

def evaluate_image_binaire(img_bin, gt, mask):
    (tn, fp, fn, tp) = confusionMatrixMasked(img_bin, gt, mask)
    if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) == 0 :
        mcc = -1
    else:
        mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)))
    d = diceMasked(img_bin, gt, mask)
    acc = (tp + tn)/(tn + fp + fn + tp)
    sp = tn / (tn + fp)
    se = tp / (tp + fn)
    metrics = {'acc': acc,'sp': sp, 'se' : se, 'mcc' : mcc, 'dice' : d}
    return metrics
