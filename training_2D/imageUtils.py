import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import time
def saveImage(array, path):
    im = Image.fromarray(array)
    im.save(path)

def readImageConvertGray(image_path):
    im =  Image.open(image_path).convert('LA')
    im = np.array(im)
    im = im[:im.shape[0], :im.shape[1], 0]
    return im

def readImage(image_path):
    im = Image.open(image_path)
    im = np.array(im)
    return im

def normalizeImage(image, max_value):
    min = np.amin(image)
    max = np.amax(image)
    n_im = ((image.astype(np.float) - min)/(max-min))*max_value
    return n_im

def normalizeImage_torch(image, max_value):
    min = torch.min(image)
    max = torch.max(image)
    n_im = ((image - min)/(max-min))*max_value
    return n_im

def denormalizeImage_torch(image, min, max):
    #image = normalizeImage_torch(image, 1)
    max_value = torch.max(image)
    min_value = torch.min(image)
    print(max_value, min_value)
    im_inv = image*(max-min)*(max_value- min_value)+min
    return im_inv


def showImage(im, title="", colormap="gray", save=""):
    plt.figure()
    plt.imshow(im, colormap)
    plt.title(title)
    plt.show()

def showImageTorch(im, title="", colormap="gray", save=""):
    plt.figure()
    plt.imshow(im.detach(), colormap)
    plt.title(title)


def removeBadFile(file_path):
    if sorted(os.listdir(file_path))[0] == ".DS_Store":
        path_image = file_path + "/.DS_Store"
        os.remove(path_image)

def EraseFile(repertoire):
    files=os.listdir(repertoire)
    if len(files) != 0:
        for i in range(0,len(files)):
            os.remove(repertoire+'/'+files[i])

def cropImages(path, xmin, xmax, ymin, ymax, test = False):
    # img_list = sorted(os.listdir(path))
    if test :
        # for i in img_list:
        imagePath = path #+i
        img = readImage(imagePath)
        img = img[xmin:xmax, ymin:ymax]
        plt.figure()
        plt.imshow(img, 'gray')
        plt.show()
    else :
        # for i in img_list:
        # print(i)
        imagePath = path #+i
        img = readImage(imagePath)
        img = img[xmin:xmax, ymin:ymax]
        name = imagePath.split('.')[0]+"_cropped.tif"
        save_path = "home/carneiro/Documents/"+name
        saveImage(img, save_path)

def cropImage(img, xmin, xmax, ymin, ymax, title = ""):
    img = img[xmin:xmax, ymin:ymax]
    # showImage(img, title)
    return img
# def cropImage(img, xmin, xmax, ymin, ymax, title):
#     img = img[xmin:xmax, ymin:ymax]
#     showImage(img, title)


def ligneProfil(line, image):
    ligne = image[line, :].copy()
    ligne_profil = image.copy()
    ligne_profil[line, :] = 0
    return ligne, ligne_profil
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(ligne_profil, cmap='gray')
    # plt.subplot(122)
    # plt.plot(ligne)