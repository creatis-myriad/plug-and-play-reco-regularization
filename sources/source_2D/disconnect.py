import numpy as np
from skimage.morphology import binary_dilation, skeletonize
from scipy.ndimage import distance_transform_bf
from skimage.filters import gaussian
from monai.transforms import MapTransform
from typing import Any, Hashable, Optional, Tuple
from monai.config import KeysCollection
from monai.utils import MAX_SEED, ensure_tuple
from skimage import morphology
from glob import glob
import torch
from PIL import Image
from monai.data import Dataset,DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord
)
import sys
sys.path.insert(0, "../../")
from sources import image_utils
from skimage.morphology import disk
# 
# class LoadPng(MapTransform):
#     def __init__(self, keys: KeysCollection) -> None:
#         self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
#         if not self.keys:
#             raise ValueError("keys must be non empty.")
#         for key in self.keys:
#             if not isinstance(key, Hashable):
#                 raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
# 
#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             d[key] = image_utils.read_image(d[key])
#         return d


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
    def __init__(self, keys: KeysCollection, label, mask = None, mean_number = 100, std_number = 50) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.label = label
        self.mask = mask
        self.mean_number = mean_number
        self.std_number = std_number


    def __call__(self, data):
        d = dict(data)
        if self.mask == None:
            mask = np.ones(d[self.label].shape)
        else:
            mask = d[self.mask]
        artefacts = create_artefacts(d[self.label], mask, self.mean_number, self.std_number)
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


def create_disconnections(groundTruth, nb_disconnection, size_max):
    """
    :param groundtruth: binary image containing blood vessel that we want to disconnect
    :param nb_disconnection: number of disconnections to create
    :param size_max: maximal size of the created disconnections
    :return: the disconnected image and the mask of the disconnections
    """
    urns = []
    distance_map = distance_transform_bf(groundTruth, 'chessboard')
    skelet = skeletonize(groundTruth)

    # Calculate the radii of the vessels
    distance_map = skelet * distance_map

    # Calculate the radius of the vessels /!\ the radii of odd and even numbers that follow are the same
    max_vessel_size = np.unique(distance_map)

    # Retrieve the coordinates according to the vessel radius
    for i in max_vessel_size[1:]:
        coords_i = np.nonzero(distance_map == i)
        if len(coords_i) != 0:
            urns.append(np.stack(coords_i, axis=-1))

    # Take the thinnest vessels (sizes 1, 2, and 3 apparently)
    urns = urns[:3]
    nb_urns = len(urns)
    proba_urns = []
    proba_urns.append(0)

    # Calculate the probability of the urns based on the number of vessel types
    for i in range(nb_urns):
        last = proba_urns[-1]
        prob_cum = last + (2 ** (nb_urns - (i + 1))) / ((2 ** nb_urns) - 1)
        proba_urns.append(prob_cum)

    # Randomly select the types of vessels to disconnect
    drawn_urns = np.random.rand(nb_disconnection)
    image = groundTruth

    # Initialization of the disconnections
    disconnections = np.zeros(image.shape)

    # Creation of each artifact (each artifact is different)
    for i in range(len(proba_urns) - 1):
        # Calculate the number of artifacts for each category of the urn
        category = (drawn_urns > proba_urns[i]) * (drawn_urns <= proba_urns[i + 1]) * 1
        number_drawn_urn = np.sum(category)

        # Randomly select the coordinates where the disconnections are located
        point_disconnect = np.random.randint(len(urns[i]), size=number_drawn_urn)
        disconnect = np.zeros(image.shape)

        # Create each artifact of the i-th category
        for j in point_disconnect:
            # Size of the vessel at the disconnection
            vessel_size = max_vessel_size[i + 1]

            # Calculate the mean size of the disconnection (not sure why)
            mean_deco_size = size_max // vessel_size
            taille_disk = abs(int(np.random.normal(mean_deco_size, scale=4)))
            nb_pix_max = np.sum(disk(taille_disk))

            dense = np.random.randint(0, 2)
            if dense == 1:
                nb_pix = abs(int(np.random.normal(nb_pix_max // 2, scale=nb_pix_max // 4)))  # * 20 initially
            else:
                nb_pix = abs(int(np.random.normal(nb_pix_max // 4, scale=nb_pix_max // 8)))  # * 10 initially

            artifact = create_artefact(disconnect, urns[i][j][0], urns[i][j][1], taille_disk, nb_pix, 1, 0.8)
            disconnect = disconnect + artifact

        image = image - disconnect
        disconnections = disconnections + disconnect

    image = (image >= 0.1) * 1
    disconnections = (disconnections >= 0.1) * 1
    return image, disconnections


def create_artefact(disconnect, pos_x, pos_y, size_disk, mean_pix, std_pix, std_gauss = 0.7):
    '''
    allow to create an artefact that can be used as an artefact or a mask for a disconnection.
      :param disconnect: image on which we want to add an artefact
      :param pos_x: coordonnate x  where we have to create the artefact
      :param pos_y: coordonnate y  where we have to create the artefact
      :param size_disk: radius of the disc to create to emulate an artefact
      :param mean_pix: mean number of pixel that is part of the artefact
      :param std_pix: standard deviation of the number of pixels that is part of the artefact
      :param std_gauss:  standard deviation of the gaussian filter applied
      :return: an image containing the artefact only
      '''

    # creation of an image to have the mask of the artefact
    image = np.zeros(disconnect.shape)
    image = disc(image, pos_x, pos_y, size_disk)
    #get the possible coordinates of the artefact
    coords = np.nonzero(image == 1)

    #calculate the number of pixel to delete
    number_pixels = int(np.random.normal(mean_pix, scale=std_pix))
    if number_pixels <= 0:
        number_pixels =1

    #calculate the position of the pixels to delete
    if number_pixels < len(coords[0]):
        random_position = np.random.randint(len(coords[0]), size=number_pixels)
        artefact = np.zeros(image.shape)
        artefact[coords[0][random_position], coords[1][random_position]] = 1
    else:
        artefact = np.zeros(image.shape)
        artefact[coords[0], coords[1]] = 1
    artefact = (gaussian(artefact, sigma = std_gauss) >0.4) * 1.0
    return artefact



def create_artefacts(image, mask, mean_number, std_number):
    '''
       :param image: image on which we want to add artefacts
       :param mask: mask of the organ where artefacts must be added
       :param mean_number: mean number of artefacts to add
       :param std_number: standard deviation number of artefacts to add
       :return: an image containing the artefacts
       '''

    artefacts = np.zeros(image.shape)

    #draw the number of artefacts to add
    number_artefact = -1
    while number_artefact < 1:
        number_artefact = int(np.random.normal(mean_number, scale=std_number))

    # we want to add artefacts around blood vessels but not "on" and inside a given mask of an organ
    area_to_alter = (binary_dilation(image, morphology.disk(2))<0.5)*1.0
    area_to_alter = area_to_alter*mask

    #coordonnates where we can add artefacts
    coord_to_alter = np.nonzero(area_to_alter)
    coord_to_alter = np.stack(coord_to_alter, axis = -1)

    # draw the size of the artefacts
    size_disk = np.random.normal(3, scale=1, size = number_artefact).tolist()
    size_disk = [int(i) for i in size_disk]

    # for each artefact
    for i in size_disk:
        #draw of the coordinates where the artefact will be added
        coordinates_nb = np.random.randint(len(coord_to_alter))
        coord_artefact = coord_to_alter[coordinates_nb]

        #creation of the artefact
        nb_pix_max = np.sum(morphology.disk(i))
        artefact = create_artefact(image, coord_artefact[0], coord_artefact[1], i, nb_pix_max // 4, nb_pix_max // 8)

        #delete the coordinate where the artefact has been added
        np.delete(coord_to_alter, coordinates_nb)

        #artefacts total
        artefacts = artefacts + artefact

    artefacts = np.clip(artefacts,0,1)
    return artefacts


###########################################################################################################################

def create_dataset(origin_directory, new_dataset_directory, nb_deco, size_deco_max, noise_level):
    images = sorted(glob(f"{origin_directory}/binary_images/*"))
    masks = sorted(glob(f"{origin_directory}/masks/*"))


    train_files = [{"image": img, "label_with_art": img,"label": img, "mask": msk} for img, msk in zip(images, masks)]
    train_trans = Compose(
        [
            LoadImaged(keys=["image", "label_with_art","label", "mask"]),
            ScaleIntensityd(keys=["image", "label_with_art", "label", "mask"]),
            BinaryDeconnect(keys=["image"], nb_deco=nb_deco, taille_max=size_deco_max),
            AddArtefacts(keys=["image", "label_with_art"], label="label", mask="mask", mean_number=noise_level, std_number=25),
            ToTensord(keys=["image", "label_with_art", "label"])
        ]
        )
    check_ds = Dataset(data = train_files, transform=train_trans)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    i = 0
    for j in range(4):
        for batch_data in check_loader:
            inputs, labels_with_art, labels = (
                batch_data["image"],
                batch_data["label_with_art"], batch_data["label"]
            )
            print(f"-------------------- Creation of the image n° {i} --------------------")

            inputs = image_utils.normalize_image(inputs.squeeze().numpy(), 255)
            inputs = np.rot90(inputs, j, [0, 1])
            Image.fromarray(inputs.astype("uint8")).save(f"{new_dataset_directory}/img_{i:d}.png")

            labels_with_art = image_utils.normalize_image(labels_with_art.squeeze().numpy(), 255)
            labels_with_art = np.rot90(labels_with_art, j, [0, 1])
            Image.fromarray(labels_with_art.astype("uint8")).save(f"{new_dataset_directory}/seg_{i:d}.png")

            fragments = labels_with_art - inputs
            Image.fromarray(fragments.astype("uint8")).save(f"{new_dataset_directory}/deco_{i:d}.png")


            labels = image_utils.normalize_image(labels.squeeze().numpy(), 255)
            labels = np.rot90(labels, j, [0, 1])
            Image.fromarray(labels.astype("uint8")).save(f"{new_dataset_directory}/label_{i:d}.png")

            boule = morphology.disk(2)
            labels_with_art = binary_dilation(fragments, boule)
            labels_with_art = image_utils.normalize_image(labels_with_art * 1.0, 255)
            Image.fromarray(labels_with_art.astype("uint8")).save(f"{new_dataset_directory}/pos_{i:d}.png")

            denoise_deconnected = ((labels - fragments) > 0.5) * 1.0 * 255
            Image.fromarray(denoise_deconnected.astype("uint8")).save(f"{new_dataset_directory}/denoise_deconnected_{i:d}.png")
            i = i + 1


