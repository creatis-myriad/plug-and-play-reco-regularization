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
            d[key],__ = create_disconnections(d[key], self.nb_deco, self.taille_max)
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
    Add a binary disc at the coordinates (x, y) with a radius of ixel_radius to an image.
    INPUTS :
    -  img: image where we add a disc
    -  x: coordinate x of the center of the disc in pixel
    -  y: coordinate y of the center of the disc in pixel
    -  pixel_radius: radius of the disc

    OUTPUT
    - img: binary image with disc added composed of 1
    '''
    for i in range(x - pixel_radius, x + pixel_radius + 1):
        r = int(np.sqrt(pixel_radius ** 2 - (i - x) ** 2))
        for j in range(y - r, y + r + 1):
            if not (i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1]):
                img[i, j] = 1
    return img

def create_disconnections(groundtruth, nb_disconnection, max_size):
    """
    Create disconnections on a binary image
    INPUTS:
        - groundtruth : binary volume with curvilinear structures to disconnect
        - nb_disconnection : number to disconnections applied on the curvilinear structure
        - max_size : maximum size applied to disconnect

    OUTPUTS:
        - image : image with disconnected curvilinear structures
        - disconnections : locations of the disconnections
    """

    urnes = []
    distance_map = distance_transform_bf(groundtruth, 'chessboard')
    skelet = skeletonize(groundtruth)

    # compute radius of curvilinear structures
    distance_map = skelet*distance_map
    vaisseau_max = np.unique(distance_map)

    # sort coordinates of the curvilinear structure centerlines depending on the radius of the curvilinear structures
    for i in vaisseau_max[1:]:
        coords_i = np.nonzero(distance_map == i)
        if len(coords_i)!=0:
            urnes.append(np.stack(coords_i, axis=-1))

    # selection of the thinnest structures
    urnes = urnes[:3]
    nb_urnes= len(urnes)
    proba_urnes = []
    proba_urnes.append(0)

    # definition of the probability to disconnect the thinnest structures
    for i in range(nb_urnes):
        last = proba_urnes[-1]
        prob_cum = last+(2**(nb_urnes-(i+1)))/((2**nb_urnes)-1)
        proba_urnes.append(prob_cum)

    # draw the curvilinear structures size to disconnect
    tirage_urnes = np.random.rand(nb_disconnection)
    image = groundtruth

    #initialisation of the disconnections
    disconnections = np.zeros(image.shape)

    # creation of each disconnection (each disconnection is different)
    for i in range(len(proba_urnes)-1):
        # compute the number of disconnections for each kind of radius of structure
        categorie = (tirage_urnes > proba_urnes[i]) * (tirage_urnes <= proba_urnes[i + 1]) * 1
        nombre_tirage_urne = np.sum(categorie)

        # draw coordinates where disconnections happened
        point_disconnect = np.random.randint(len(urnes[i]), size=nombre_tirage_urne)
        disconnect = np.zeros(image.shape)

        #compute each disconnection of the i-th category
        for j in point_disconnect:
            #size of the curvilinear structures at the disconnection point
            taille_vaisseau = vaisseau_max[i+1]

            #compute the size of the mean disconnection
            taille_deco_mean = max_size // taille_vaisseau
            taille_disk = abs(int(np.random.normal(taille_deco_mean, scale=4)))

            nb_pix_max = np.sum(disk(taille_disk))

            # kind of disconnections
            dense = np.random.randint(0, 2)
            if dense == 1:
                nb_pix = abs(int(np.random.normal(nb_pix_max // 2, scale=nb_pix_max//4)))
            else:
                nb_pix = abs(int(np.random.normal(nb_pix_max // 4, scale=nb_pix_max//8)))

            #creation of the disconnection
            tache = create_tache(disconnect, urnes[i][j][0], urnes[i][j][1], taille_disk, nb_pix, 1, 0.8)
            disconnect = disconnect + tache
        image = image - disconnect
        disconnections = disconnections + disconnect
    image = (image >= 0.1) * 1
    disconnections = (disconnections >= 0.1) * 1
    return image, disconnections


def create_tache(disconnect, pos_x, pos_y, size_disk, mean_pix, std_pix, std_gauss = 0.7):
    """
       Create a disconnection on a binary image
       INPUTS:
           - disconnect : image that contains disconnections
           - pos_x : position x of the future disconnetion
           - pos_y : position y of the future disconnetion
           - size_disk : size of the disconnection possible
           - mean_pix : mean of the number of pixels that we want to disconnect
           - std_pix : Standard deviation the number of pixels that we want to disconnect
           - std_gauss : gaussian smoothness

       OUTPUTS:
           - disconnection :
       """

    # creation of an image with the mask of the disconnection
    image = np.zeros(disconnect.shape)
    image = disc(image, pos_x, pos_y, size_disk)
    #coordinates possible inside the mask
    coords_1 = np.nonzero(image == 1)

    # compute the number of pixel that composed the disconnection
    nombre_pixels = int(np.random.normal(mean_pix, scale=std_pix))
    if nombre_pixels <= 0:
        nombre_pixels =1
    if nombre_pixels < len(coords_1[0]):
        pos_aleatoire_1 = np.random.randint(len(coords_1[0]), size=nombre_pixels)
        disconnection = np.zeros(image.shape)
        disconnection[coords_1[0][pos_aleatoire_1], coords_1[1][pos_aleatoire_1]] = 1
    else :
        disconnection = np.zeros(image.shape)
        disconnection[coords_1[0], coords_1[1]] = 1
    disconnection = (gaussian(disconnection, sigma = std_gauss) >0.4) * 1.0

    return disconnection



def create_mapTaches(image, mask, mean_taches, std_taches):
    """
       Create a map of artefact avoiding the curvilinear structure. The number of artefact created follow a normal law with a mean
       of mean_taches and a standard deviation of std_taches
       INPUTS:
           - image : image to add artefacts
           - mask : mask of the curvilinear structure where no artefact must be adding
           - mean_taches : mean number of artefact to add
           - std_taches : standard deviation used to draw the number of artefact to create

       OUTPUTS:
           - taches : map of artefact
       """


    taches = np.zeros(image.shape)

    #draw the number of artefacts to create
    nombre_taches = -1
    while nombre_taches < 1:
        nombre_taches = int(np.random.normal(mean_taches, scale=std_taches))

    # add artefacts around curvilinear structures
    zones_to_taches = (binary_dilation(image, disk(2)) < 0.5) * 1.0
    zones_to_taches = zones_to_taches * mask

    #coordinates where we can put artefacts
    coord_to_tache = np.nonzero(zones_to_taches)
    coord_to_tache = np.stack(coord_to_tache, axis = -1)

    # size of artefacts drawn
    taille_disk = np.random.normal(3, scale=1, size = nombre_taches).tolist()
    taille_disk = [int(i) for i in taille_disk]

    #for each artefact
    for i in taille_disk:
        # draw the coordinate
        num_pix_to_tache = np.random.randint(len(coord_to_tache))
        coord_tache = coord_to_tache[num_pix_to_tache]

        #create artefact
        nb_pix_max = np.sum(disk(i))
        tache = create_tache(image, coord_tache[0], coord_tache[1], i, nb_pix_max // 4, nb_pix_max // 8)

        # delete coordinates
        np.delete(coord_to_tache, num_pix_to_tache)

        # add artefact with others
        taches = taches + tache
    # clip to have a binary image
    taches = np.clip(taches,0,1)
    return taches

################################### Ponderated loss ##########################################################
class PonderatedDiceloss(nn.Module):
    """
    ponderated dice loss to accentute loss on the disconnections
        INPUTS:
            - input : image that contains disconnections
            - target : groundtruth
            - mask : mask of the disconnection
        OUTPUTS:
            - dice : global dice
            - dice_1 : Dice on the global structure
            - dice_2 : dice on the disconnections
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