import numpy as np
from skimage.morphology import skeletonize, binary_erosion, ball
from scipy import ndimage
import imageUtils
from monai.transforms import MapTransform
from typing import Any, Hashable, Optional, Tuple
from monai.config import KeysCollection
from monai.utils import MAX_SEED, ensure_tuple
from skimage.measure import label
from monai.data import write_nifti
import torch
from torch import nn
import torch.nn.functional as F
# ################################### Fonctions pour déconnexions et artefacts ###########################################

def cube(img, x, y, z, pixel_radius):
    '''
    :param img: image ou on doit ajouter un cube
    :param x: coordonnee x du centre du cube a ajouter
    :param y: coordonnee y du centre du cube a ajouter
    :param z: coordonnee z du centre du cube a ajouter
    :param pixel_radius: rayon du cube
    :return: l'image avec un cube dedans au coordonnees x, y, z de rayon pixel_radius
    '''
    for i in range(x - pixel_radius, x + pixel_radius + 1):
        for j in range(y - pixel_radius, y + pixel_radius + 1):
            for k in range(z - pixel_radius, z + pixel_radius + 1):
                if not (i < 0 or j < 0 or k < 0 or i >= img.shape[0] or j >= img.shape[1] or  k >= img.shape[2]):
                    img[i, j, k] = 1
    return img


def cube_vide(img, x, y, z, pixel_radius):
    img = cube(np.zeros(img.shape), x, y, z, pixel_radius) - cube(np.zeros(img.shape), x, y, z, pixel_radius - 1)
    return img


def boule(img, x, y, z, pixel_radius):
    '''
    :param img: image ou on doit ajouter un cube
    :param x: coordonnee x du centre du cube a ajouter
    :param y: coordonnee y du centre du cube a ajouter
    :param z: coordonnee z du centre du cube a ajouter
    :param pixel_radius: rayon du cube
    :return: l'image avec un cube dedans au coordonnees x, y, z de rayon pixel_radius
    '''
    print(x, y, z,pixel_radius, int(pixel_radius))
    pixel_radius = int(pixel_radius)
    for i in range(x - pixel_radius, x + pixel_radius + 1):
        for j in range(y - pixel_radius, y + pixel_radius +1):
            for k in range(z - pixel_radius, z + pixel_radius +1):
                distance = np.sqrt((i-pixel_radius-x)**2 + (j-pixel_radius-y)**2 + (k-pixel_radius-z)**2)
                if distance <= pixel_radius and  not (i < 0 or j < 0 or k < 0 or i >= img.shape[0] or j >= img.shape[1] or  k >= img.shape[2]):
                    img[i,j,k] = 1
    return img

def create_deconnexion_simple(image, skelet,  distance_map, x, y, z, r):

    # deco cubique
    # deco =np.ones([r, r, r])
    disconnect = np.zeros(image.shape)
    disconnect = cube(disconnect, x, y, z, r//2)
    # calcul du bord de la deconnexion pour un cube
    # localisation du bord dans l'image
    bord_deco_im = np.zeros(image.shape)

    bord_deco_im = cube_vide(bord_deco_im, x, y, z, r//2)
    print("test", np.sum(bord_deco_im), np.sum(disconnect))
    # recuperation des bords de vaisseaux si existants
    bords_vaisseaux = skelet * bord_deco_im
    print("intersection", np.sum(bords_vaisseaux))
    coords_i = np.nonzero(bords_vaisseaux)
    coords_i = np.stack(coords_i, axis=-1)
    print(coords_i)
    # nombre de bout

    modelisation_extremite = np.zeros(image.shape)
    # recuperation rayon des extremites et fixation des boules de bout
    # recuperation des coordonnees des 1 ou 2 points :)

    for coord in coords_i:
        rayon = distance_map[coord[0]][coord[1]][coord[2]]
        ball_1 = boule(np.zeros(image.shape), coord[0], coord[1], coord[2], rayon)
        modelisation_extremite += ball_1.copy()
    modelisation_extremite = (modelisation_extremite >= 1) * 1

    disconnect = ((disconnect - modelisation_extremite) == 1) * 1
    return disconnect

################################################## Déconnexions binaires ###############################################
def createDisconnections(image, nb_disconnection, taille_max_deco = 8, nb_val_ray = None):
    urnes = []
    image = imageUtils.normalizeImage(image, 1)

    # recuperation du squelette du réseau vasculaire
    skelet = imageUtils.normalizeImage(skeletonize(image), 1)

    # recuperation du rayon des vaisseaux au niveau du squelette
    distance_map = ndimage.distance_transform_bf(image, 'chessboard')
    distance_map = skelet * distance_map

    # calcul du rayon des vaisseaux  /!\ les rayons des nombres impaire et paires qui se suivent sont le meme
    rayons_vaisseaux = np.unique(distance_map)
    # print(rayons_vaisseaux)

    # recuperation des coordonnées selon le rayon du vaisseau
    for i in rayons_vaisseaux[1:]:
        coords_i = np.nonzero(distance_map == i)
        if len(coords_i) != 0:
            urnes.append(np.stack(coords_i, axis=-1))

    # calcul proba des urnes selon le nombre de type de vaisseau
    if nb_val_ray == None:
        nb_urnes = len(urnes)
    else :
        nb_urnes = min(len(urnes), nb_val_ray)
    proba_urnes = []
    proba_urnes.append(0)
    for i in range(nb_urnes):
        last = proba_urnes[-1]
        prob_cum = last + (2 ** (nb_urnes - (i + 1))) / ((2 ** nb_urnes) - 1)
        proba_urnes.append(prob_cum)

    print("probas:", proba_urnes)
    # tirage au sort des types de vaisseaux déconnectés
    tirage_urnes = np.random.rand(nb_disconnection)
    print("tirage_urnes:", tirage_urnes)

    image_deco = image

    #initialisation des deconnections
    disconnections = np.zeros(image.shape)

    # creation de chaque tache (chaque tache est differente)
    for i in range(len(proba_urnes) - 1):
        # calcul du nombre de taches pour chaque categorie de l'urne
        categorie = (tirage_urnes > proba_urnes[i]) * (tirage_urnes <= proba_urnes[i + 1]) * 1
        nombre_tirage_urne = np.sum(categorie)

        # tirage aux sort des coordonnées ou sont situés les deconnexions
        point_disconnect = np.random.randint(len(urnes[i]), size=nombre_tirage_urne)

        #creation de chaque tache de la ieme categorie
        for j in point_disconnect:
            # parametres de la deco
            taille_vaisseau = rayons_vaisseaux[i + 1]
            taille_deco_mean = taille_max_deco // taille_vaisseau
            taille_disk = abs(int(np.random.normal(taille_deco_mean, scale=2)))
            # taille_disk = int(taille_deco_mean)
            if taille_disk == 0:
                taille_disk = 1
            print("taille disk:", taille_disk)
            # position du point du vaisseau a deco
            x = urnes[i][j][0]
            y = urnes[i][j][1]
            z = urnes[i][j][2]
            # creation de la deconnexion
            disconnect = create_deconnexion_simple(image, skelet, distance_map, x, y, z, taille_disk)

            # deconnexion du vaisseau selectionne
            image_deco = image_deco - disconnect
            image_deco = (image_deco == 1) * 1

            #masque de deconnection total
            disconnections = disconnections + disconnect
            disconnections = (disconnections >= 1) * 1

    return image_deco,  disconnections


def generatorNoise(image_in ,noise_size,  seuil):
    #recuperation de la taille de l'image ou il faut rajouter le bruit
    size = image_in.shape

    # creation d'un bruit uniforme
    s = np.random.uniform(-1, 1, size)

    # transformée de fourier
    S = np.fft.fftn(s)

    #creation d'un filtre passe bas
    high_frequencies = np.zeros(s.shape)
    filter = ball(noise_size)
    high_frequencies[0:noise_size + 1, 0:noise_size + 1, 0:noise_size + 1] = filter[noise_size:, noise_size:, noise_size:]

    # application du filtre
    S_filt = S * high_frequencies

    # transformée inverse
    s_inv = np.fft.ifftn(S_filt)

    # normalisation pour seuiller
    s_inv = imageUtils.normalizeImage(s_inv, 1)

    # seuillage et fusion des deux images
    image = (s_inv > seuil) * 1.0
    image = image + image_in
    image = (image > 0)

    # recherche de la plus grande componante
    regions = label(image, connectivity=2)
    counts = []
    counts.append(0)
    nb_composante_init = np.max(label(image_in, connectivity=2))

    #recherche des labels les plus grands (contenant reseau vasculaire et le bruit colle dessus)
    for i in range(1, np.amax(regions) + 1):
        count = np.count_nonzero(regions == i)
        counts.append(count)

    #recolte des composantes du réseau vasculaire initial (les plus grandes dans l'image bruitée)
    counts_max = sorted(counts)[-nb_composante_init:]
    vascular_indices = []
    for i in counts_max:
        vascular_indices.append(counts.index(i))

    # soustraction pour enlever les bruits collés aux vaisseaux
    for vascular_indice in vascular_indices:
        image = image * 1.0 - (regions == vascular_indice) * 1.0
    # refusion de l'image et deconnectee
    image = (image > 0) * 1.0
    return image

################################################## Déconnexions binaires ###############################################
def createDisconnections_2(image, nb_disconnection, taille_max_deco = 8):
    urnes = []

    image = imageUtils.normalizeImage(image, 1)

    # recuperation du squelette du réseau vasculaire
    skelet = imageUtils.normalizeImage(skeletonize(image), 1)
    print("debut de la distance map magueule")
    # recuperation du rayon des vaisseaux au niveau du squelette
    distance_map = ndimage.distance_transform_bf(image, 'chessboard')
    print("fin de la distance map magueule")

    distance_map = skelet * distance_map

    # calcul du rayon des vaisseaux  /!\ les rayons des nombres impaire et paires qui se suivent sont le meme
    rayons_vaisseaux = np.unique(distance_map)
    print(rayons_vaisseaux)

    # recuperation des coordonnées selon le rayon du vaisseau
    for i in rayons_vaisseaux[1:]:
        coords_i = np.nonzero(distance_map == i)
        if len(coords_i) != 0 and i == 1:
           urnes.append(np.stack(coords_i, axis=-1))
        elif len(coords_i) != 0 and i==2:
            other_coord = np.stack(coords_i, axis=-1)
        else:
            other_coord = np.concatenate((other_coord, np.stack(coords_i, axis=-1)))
    urnes.append(other_coord)
    print(len(urnes))
    print("fin de la recuperation des coordonnées selon le rayon du vaisseau")

    # calcul proba des urnes selon le nombre de type de vaisseau
    # nb_urnes =4
    proba_urnes = [0, 0.95,  1]
    #
    print("probas:", proba_urnes)

    # tirage au sort des types de vaisseaux déconnectés
    tirage_urnes = np.random.rand(nb_disconnection)
    print("tirage_urnes:", tirage_urnes)

    image_deco = image
    disconnections = np.zeros(image.shape)
    for i in range(len(proba_urnes) - 1):
        # recuperation des tirages de la catégorie traitée
        categorie = (tirage_urnes > proba_urnes[i]) * (tirage_urnes <= proba_urnes[i + 1]) * 1

        # compter le nombre de déconnexion pour cette categorie
        nombre_tirage_urne = np.sum(categorie)

        # tirage aux sort des coordonnées ou sont situés les deconnexions
        print(len(urnes[i]), nombre_tirage_urne)
        point_disconnect = np.random.randint(len(urnes[i]), size=nombre_tirage_urne)

        for j in point_disconnect:
            # parametres de la deco
            taille_vaisseau = rayons_vaisseaux[i + 1]
            taille_deco_mean = taille_max_deco // taille_vaisseau
            taille_disk = abs(int(np.random.normal(taille_deco_mean, scale=2)))
            # taille_disk = int(taille_deco_mean)
            if taille_disk == 0:
                taille_disk = 1
            print("taille disk:", taille_disk)
            # position du point du vaisseau a deco
            x = urnes[i][j][0]
            y = urnes[i][j][1]
            z = urnes[i][j][2]
            # creation de la deconnexion
            disconnect = create_deconnexion_simple(image, skelet, distance_map, x, y, z, taille_disk)

            # deconnexion du vaisseau selectionne
            image_deco = image_deco - disconnect
            image_deco = (image_deco == 1) * 1

            #masque de deconnection total
            disconnections = disconnections + disconnect
            disconnections = (disconnections >= 1) * 1

    return image_deco,  disconnections

####################################################### Transformée de deconnection #####################################
class BinaryDeconnect(MapTransform):
    def __init__(self, keys: KeysCollection,  nb_deco, taille_max_deco = 8, nb_val_ray=None) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.nb_deco = nb_deco
        self.taille_max_deco = taille_max_deco
        self.nb_val_ray = nb_val_ray
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key],__ = createDisconnections(d[key], self.nb_deco, self.taille_max_deco, self.nb_val_ray)
        return d


class Binaries(MapTransform):
    def __init__(self, keys: KeysCollection, value) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.value = value
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > self.value)*1
        return d


class AddArtefacts(MapTransform):
    def __init__(self, keys: KeysCollection, label,  mean_taches = 20, seuil = 0.8) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.label = label
        self.mean_taches = mean_taches
        self.seuil = seuil


    def __call__(self, data):
        d = dict(data)
        artefacts = generatorNoise(d[self.label], self.mean_taches, self.seuil)
        for key in self.keys:
            d[key] = d[key] + artefacts
            d[key] = np.clip(d[key], 0, 1)
        return d

class BinaryDeconnect_2(MapTransform):
    def __init__(self, keys: KeysCollection,  nb_deco, taille_max_deco = 8) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.nb_deco = nb_deco
        self.taille_max_deco = taille_max_deco

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key],__ = createDisconnections_2(d[key], self.nb_deco, self.taille_max_deco)
        return d



##################################### Loss function ####################################################################

class PrecisionLoss(nn.Module):
    """Criterion Precision loss for binary classification

     Shape:
        - Input: b * H * W * Z
        - Target:b * H * W * Z
    """

    def __init__(self) -> None:
        super(PrecisionLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.shape == target.shape:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))

        # compute the actual dice score
        if len(input.shape) == 5: ### ca pue mais ca marche
            dim = [1, 2, 3, 4]
        elif len(input.shape) == 4:
            dim = [1, 2, 3]

        tp = torch.sum(input * target, dim = dim)# TP
        denomin = torch.sum(input, dim = dim) # TP + FP (mais flou)
        precision = tp / (denomin + self.eps)
        # print("avant moyenne", precision.shape)
        moyenne = torch.mean(1. - precision)
        # print("apres moyenne", moyenne.shape)

        return moyenne

# # Dice loss for Pytorch
# def dice_loss_pytorch(y_pred, y_true):
#     epsilon = 1e-5
#     intersection = torch.sum(y_pred * y_true, dim=list(range(1, y_pred.dim())))
#     union = torch.sum(y_pred, dim=list(range(1, y_pred.dim()))) + torch.sum(y_true, dim=list(range(1, y_true.dim())))
#     return torch.mean(1.0 - (2. * intersection + epsilon) / (union + epsilon))


class PonderatedDiceloss(nn.Module):
    """Criterion Precision loss for binary classification

     Shape:
        - Input: b * H * W * Z
        - Target:b * H * W * Z
    """

    def __init__(self) -> None:
        super(PonderatedDiceloss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
            ) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.shape == target.shape:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))

        intersection_1 = torch.sum(input * target, dim=list(range(1, input.dim())))
        union_1 = torch.sum(input, dim=list(range(1, input.dim()))) + torch.sum(target,
                                                                                dim=list(range(1, target.dim())))
        dice_1 = torch.mean(1.0 - (2. * intersection_1 + self.eps) / (union_1 + self.eps))

        target_2 = target * mask
        intersection_2 = torch.sum(input * mask * target_2, dim=list(range(1, input.dim())))
        union_2 = torch.sum(input * mask, dim=list(range(1, input.dim()))) + torch.sum(target_2,
                                                                                dim=list(range(1, target_2.dim())))
        dice_2 = torch.mean(1.0 - (2. * intersection_2 + self.eps) / (union_2 + self.eps))
        dice = dice_1 + dice_2
        return dice, dice_1, dice_2

################################### Fonctions pour déconnexions et artefacts ###########################################
# # import nibabel as ni
# from glob import glob
# import os
# # img_file = "/home/carneiro/Documents/datas/BraVa/GT"
# # gts = sorted(glob(os.path.join(img_file, "MRA*")))
# # for image_dir_path in gts:
# #     dir = ni.load(image_dir_path).get_fdata()
# #     image_deco, disconnections = createDisconnections_2(dir, 100)
# #     write_nifti(data=image_deco, file_name=f"image_deco_test.nii.gz", resample=False)
# #     write_nifti(data=disconnections, file_name=f"disconnections_test.nii.gz", resample=False)
# #     break
#
# import nibabel as ni
# # img_file = "/home/carneiro/Documents/datas/BraVa/GT/MRA1_GT.nii.gz"
# img_file = "/home/carneiro/Documents/datas/BraVa/GT"
# gts = sorted(glob(os.path.join(img_file, "MRA*")))
# dir = ni.load(gts[0]).get_fdata()
# img_deco_file = "image_deco_test.nii.gz"
# dir_deco = ni.load(img_deco_file).get_fdata()
# disconnections = dir - dir_deco
# write_nifti(data=disconnections > 0, file_name=f"disconnections_belles_test.nii.gz", resample=False)

# ############################ test sur un parallelipede rectangle
#
# tube = np.zeros([10, 10, 10])
# tube[3:6, :, 3:6] = 1
# x, y, z, r = 9, 7, 9, 5
# taille_max_deco = r
#
#
# image = np.zeros(tube.shape + np.array([2 * taille_max_deco, 2 * taille_max_deco, 2 * taille_max_deco]))
#
# image[image.shape[0]//2 - tube.shape[0]//2 : image.shape[0]//2 + tube.shape[0]//2 + tube.shape[0] % 2,
#       image.shape[1]//2 - tube.shape[1]//2 : image.shape[1]//2 + tube.shape[1]//2 + tube.shape[1] % 2,
#       image.shape[2]//2 - tube.shape[2]//2 : image.shape[2]//2 + tube.shape[2]//2 + tube.shape[2] % 2] = tube
#
# image = imageUtils.normalizeImage(image, 1)
# write_nifti(data=image, file_name=f"test.nii.gz", resample=False)
#
# # recuperation du squelette du réseau vasculaire
# skelet = imageUtils.normalizeImage(skeletonize(image), 1)
# write_nifti(data=skelet, file_name=f"skelet.nii.gz", resample=False)
#
# print(skelet)
# # recuperation du rayon des vaisseaux au niveau du squelette
# distance_map = ndimage.distance_transform_bf(image, 'chessboard')
# distance_map = skelet * distance_map
#
# disconnect = create_deconnexion_simple(image, skelet,  distance_map,x, y, z, r)
# write_nifti(data=disconnect, file_name=f"disconnect.nii.gz", resample=False)
#
# # deconnexion du vaisseau selectionne
# image_deco = image - disconnect
# image_deco = (image_deco == 1) * 1
#
# write_nifti(data=image_deco, file_name=f"image_deco.nii.gz", resample=False)
# # write_nifti(data=ball(3), file_name=f"b_3.nii.gz", resample=False)
# # write_nifti(data=ball(4), file_name=f"b_4.nii.gz", resample=False)
