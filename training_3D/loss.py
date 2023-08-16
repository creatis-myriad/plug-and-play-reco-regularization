import numpy as np
import torch.nn as nn
import monai
import torch
from skimage.measure import label


def loss_with_connex_number(loss, val_outputs, val_labels, alpha = 0.000005):
    loss_numb = loss(val_outputs, val_labels)
    label_connex_output = label(val_outputs.squeeze().detach().numpy(),connectivity=2)
    ratio_connex_numb = np.abs(np.amax(label_connex_output) - np.amax(label(val_labels.squeeze().detach().numpy(),connectivity=2)))
    print("lolé", np.amax(label(val_labels.squeeze().detach().numpy(),connectivity=2)))
    return loss_numb + alpha * ratio_connex_numb