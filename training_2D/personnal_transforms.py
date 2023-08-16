import numpy as np
from skimage.morphology import disk, binary_dilation, skeletonize
from scipy.ndimage import distance_transform_bf
from skimage.filters import gaussian
from monai.transforms import MapTransform
from typing import Any, Hashable, Optional, Tuple
from monai.config import KeysCollection
from monai.utils import MAX_SEED, ensure_tuple
import torch
from torch import nn


####################################################### Transformée de deconnection #######################################
class BinaryDeconnect(MapTransform):
    def __init__(self, keys: KeysCollection,  nb_deco, taille_max) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.nb_deco = nb_deco
        self.taille_max = taille_max

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key],__ = createDisconnections(d[key], self.nb_deco, self.taille_max)
        return d

class AddArtefacts(MapTransform):
    def __init__(self, keys: KeysCollection, label, mask = None,  mean_taches = 100, std_taches = 50) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.label = label
        self.mask = mask
        self.mean_taches = mean_taches
        self.std_taches = std_taches


    def __call__(self, data):
        d = dict(data)
        if self.mask == None:
            masked = np.ones(d[self.label].shape)
        else:
            masked = d[self.mask]
        artefacts = create_mapTaches(d[self.label], masked, self.mean_taches, self.std_taches)
        for key in self.keys:
            d[key] = d[key] + artefacts
            d[key] = np.clip(d[key], 0, 1)
        return d



################################################## Déconnexions binaires #############################################

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

def createDisconnections(groundTruth, nb_disconnection,  taille_max):
    # taille_max = 17 # 12 initialement
    urnes = []
    distance_map = distance_transform_bf(groundTruth, 'chessboard')
    skelet = skeletonize(groundTruth)

    # calcul des rayons des vaisseaux
    distance_map = skelet*distance_map

    # calcul du rayon des vaisseaux  /!\ les rayons des nombres impaire et paires qui se suivent sont le meme
    vaisseau_max = np.unique(distance_map)

    # recuperation des coordonnées selon le rayon du vaisseau
    for i in vaisseau_max[1:]:
        coords_i = np.nonzero(distance_map == i)
        if len(coords_i)!=0:
            urnes.append(np.stack(coords_i, axis=-1))

    # on prend les vaisseaux les plus fins ( de tailles 1 2 et 3 apparemment)
    urnes = urnes[:3]
    nb_urnes= len(urnes)
    proba_urnes = []
    proba_urnes.append(0)

    # calcul proba des urnes selon le nombre de type de vaisseau
    for i in range(nb_urnes):
        last = proba_urnes[-1]
        prob_cum = last+(2**(nb_urnes-(i+1)))/((2**nb_urnes)-1)
        proba_urnes.append(prob_cum)

    # tirage au sort des types de vaisseaux déconnectés
    tirage_urnes = np.random.rand(nb_disconnection)
    image = groundTruth

    #initialisation des deconnections
    disconnections = np.zeros(image.shape)

    # creation de chaque tache (chaque tache est differente)
    for i in range(len(proba_urnes)-1):
        # calcul du nombre de taches pour chaque categorie de l'urne
        categorie = (tirage_urnes > proba_urnes[i]) * (tirage_urnes <= proba_urnes[i + 1]) * 1
        nombre_tirage_urne = np.sum(categorie)

        # tirage aux sort des coordonnées ou sont situés les deconnexions
        point_disconnect = np.random.randint(len(urnes[i]), size=nombre_tirage_urne)
        disconnect = np.zeros(image.shape)

        #creation de chaque tache de la ieme categorie
        for j in point_disconnect:
            #taille du vaisseau au niveau de la deconnexion
            taille_vaisseau = vaisseau_max[i+1]

            #calcul de la taille de la deconexion moyenne (pourquoi ca je sais plus)
            taille_deco_mean = taille_max // taille_vaisseau
            taille_disk = abs(int(np.random.normal(taille_deco_mean, scale=4)))

            nb_pix_max = np.sum(disk(taille_disk))

            dense = np.random.randint(0, 2)
            if dense == 1:
                nb_pix = abs(int(np.random.normal(nb_pix_max // 2, scale=nb_pix_max//4))) # * 20 initialement
            else:
                nb_pix = abs(int(np.random.normal(nb_pix_max // 4, scale=nb_pix_max//8))) # * 10 initialement
            tache = create_tache_2(disconnect, urnes[i][j][0], urnes[i][j][1],taille_disk, nb_pix, 1, 0.8)
            disconnect = disconnect + tache
        image = image - disconnect
        disconnections = disconnections + disconnect
    image = (image >= 0.1) * 1
    disconnections = (disconnections >= 0.1) * 1
    return image, disconnections


def create_tache_2(disconnect, pos_x, pos_y, taille_disk, mean_pix, std_pix, std_gauss = 0.7):

    # creation dune image pour avoir le mask de la tache
    image = np.zeros(disconnect.shape)
    image = disc(image, pos_x, pos_y, taille_disk)
    #recuperation des coordonnees de la tache possible (un disk)
    coords_1 = np.nonzero(image == 1)

    #calcul du nombre de pixel qu'on va niquer
    nombre_pixels = int(np.random.normal(mean_pix, scale=std_pix))

    # au cas ou on a un nombre negatif .. on sait jamais tahu
    if nombre_pixels <= 0:
        nombre_pixels =1

    #calcul des position des pixels a niquer
    if nombre_pixels < len(coords_1[0]):
        pos_aleatoire_1 = np.random.randint(len(coords_1[0]), size=nombre_pixels)
        tache = np.zeros(image.shape)
        tache[coords_1[0][pos_aleatoire_1], coords_1[1][pos_aleatoire_1]] = 1
    else :
        tache = np.zeros(image.shape)
        tache[coords_1[0], coords_1[1]] = 1
    tache = (gaussian(tache, sigma = std_gauss) >0.4) * 1.0
    return tache



def create_mapTaches(image, mask, mean_taches, std_taches):

    #creation du mask d'artefacts a retourner
    taches = np.zeros(image.shape)

    #tirage au sort du nombre d'artefacts
    nombre_taches = -1
    while nombre_taches < 1:
        nombre_taches = int(np.random.normal(mean_taches, scale=std_taches))

    # on rajoute des artefacts autour des vaisseaux mais pas dessus et a linterieur du fond de retine
    zones_to_taches = (binary_dilation(image, disk(2))<0.5)*1.0
    zones_to_taches = zones_to_taches*mask

    #coordonnees possibles ou on peut tacher
    coord_to_tache = np.nonzero(zones_to_taches)
    coord_to_tache = np.stack(coord_to_tache, axis = -1)

    # tailles des artefacts que l'on va faire tiré au sort
    taille_disk = np.random.normal(3, scale=1, size = nombre_taches).tolist()
    taille_disk = [int(i) for i in taille_disk]

    #parcours de chaque artefact
    for i in taille_disk:
        #tirage au sort de la coordonnee ou creer l'artefact
        num_pix_to_tache = np.random.randint(len(coord_to_tache))
        coord_tache = coord_to_tache[num_pix_to_tache]

        #creation de l'artefact
        nb_pix_max = np.sum(disk(i))
        tache = create_tache_2(image, coord_tache[0], coord_tache[1], i, nb_pix_max//4, nb_pix_max//8)

        #suppression de la coordonnee deja artefactee
        np.delete(coord_to_tache, num_pix_to_tache)

        #ajout de notre taches aux autres taches
        taches = taches + tache
    #on clip pour avoir une image binaire (oui oui cest con mais ca marche car soit a 0, soit a 1, soit a plus)
    taches = np.clip(taches,0,1)
    return taches
################################### Ponderated loss ##########################################################
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