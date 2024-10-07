import numpy as np
from skimage.morphology import skeletonize, ball, binary_dilation
from scipy import ndimage
from sources import image_utils
from monai.transforms import MapTransform
from typing import Any, Hashable, Optional, Tuple
from monai.config import KeysCollection
from monai.utils import MAX_SEED, ensure_tuple
from skimage.measure import label
from glob import glob
from monai.data import Dataset, DataLoader, write_nifti
import torch
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
)

# ################################### Fonctions pour déconnexions et artefacts ###########################################

def cube(img, x, y, z, pixel_radius):
    '''
   :param img: source_2D where a cube needs to be added
   :param x: x-coordinate of the center of the cube to be added
   :param y: y-coordinate of the center of the cube to be added
   :param z: z-coordinate of the center of the cube to be added
   :param pixel_radius: radius of the cube
   :return: the source_2D with a cube added at coordinates x, y, z with the specified pixel_radius
    '''
    for i in range(x - pixel_radius, x + pixel_radius + 1):
        for j in range(y - pixel_radius, y + pixel_radius + 1):
            for k in range(z - pixel_radius, z + pixel_radius + 1):
                if not (i < 0 or j < 0 or k < 0 or i >= img.shape[0] or j >= img.shape[1] or  k >= img.shape[2]):
                    img[i, j, k] = 1
    return img


def empty_cube(img, x, y, z, pixel_radius):
    '''
   :param img: source_2D where an empty cube needs to be added
   :param x: x-coordinate of the center of the empty cube to be added
   :param y: y-coordinate of the center of the empty cube to be added
   :param z: z-coordinate of the center of the empty cube to be added
   :param pixel_radius: radius of the empty cube
   :return: the source_2D with an empty cube added at coordinates x, y, z with the specified pixel_radius
    '''
    img = cube(np.zeros(img.shape), x, y, z, pixel_radius) - cube(np.zeros(img.shape), x, y, z, pixel_radius - 1)
    return img


def custom_ball(img, x, y, z, pixel_radius):
    '''
   :param img: source_2D where a ball needs to be added
   :param x: x-coordinate of the center of the ball to be added
   :param y: y-coordinate of the center of the ball to be added
   :param z: z-coordinate of the center of the ball to be added
   :param pixel_radius: radius of the ball
   :return: the source_2D with a ball added at coordinates x, y, z with the specified pixel_radius
    '''
    pixel_radius = int(pixel_radius)
    for i in range(x - pixel_radius, x + pixel_radius + 1):
        for j in range(y - pixel_radius, y + pixel_radius +1):
            for k in range(z - pixel_radius, z + pixel_radius +1):
                distance = np.sqrt((i-pixel_radius-x)**2 + (j-pixel_radius-y)**2 + (k-pixel_radius-z)**2)
                if distance <= pixel_radius and not (i < 0 or j < 0 or k < 0 or i >= img.shape[0] or j >= img.shape[1] or  k >= img.shape[2]):
                    img[i,j,k] = 1
    return img

def create_simple_deconnexion(image, skelet, distance_map, x, y, z, r):
    """
   :param image: source_2D containing the vascular structure that we want to disconnect
   :param skelet: centerlines of the vascular structure
   :param distance_map: distance map of the source_2D
   :param x: x-coordinate of the center of the disconnection to add
   :param y: y-coordinate of the center of the disconnection to add
   :param z: z-coordinate of the center of the disconnection to add
   :param r: radius of the disconnection
    return the source_2D containing the disconnected vascular structure
    """
    disconnect = np.zeros(image.shape)
    disconnect = cube(disconnect, x, y, z, r//2)

    # calculation of the disconnection edge for a cube
    # locating the edge within the source_2D
    edge_disconnection = np.zeros(image.shape)
    edge_disconnection = empty_cube(edge_disconnection, x, y, z, r // 2)

    # getting the vessel edges if they exist
    edge_vessels = skelet * edge_disconnection
    coords_i = np.nonzero(edge_vessels)
    coords_i = np.stack(coords_i, axis=-1)

    endings = np.zeros(image.shape)

    # getting the radius of the extremities and setting the end spheres
    # getting the coordinates of the different points
    for coord in coords_i:
        rayon = distance_map[coord[0]][coord[1]][coord[2]]
        ball_1 = custom_ball(np.zeros(image.shape), coord[0], coord[1], coord[2], rayon)
        endings += ball_1.copy()
    endings = (endings >= 1) * 1
    disconnect = ((disconnect - endings) == 1) * 1

    return disconnect

################################################## Déconnexions binaires ###############################################
def create_disconnections(image, nb_disconnection, size_max_deco = 8, nb_val_rad = None):
    """
   :param image: source_2D containing the vascular structure that we want to disconnect
   :param nb_disconnection: number of disconnection to add to vascular structure
   :param size_max_deco: mean maximal size of disconnection that can be applied
   :param nb_val_rad: number of centerline types that can be disconnected
    return the source_2D containing the disconnected vascular structure
    """

    urns = []
    image = image_utils.normalize_image(image, 1)

    # getting the centerlines of the vascular network
    skelet = image_utils.normalize_image(skeletonize(image), 1)

    # getting the radius of the blood vessel on the centerline
    distance_map = ndimage.distance_transform_bf(image, 'chessboard')
    distance_map = skelet * distance_map
    radius_vessels = np.unique(distance_map)

    # getting the coordinates of the centerlines linked to a radius value
    for i in radius_vessels[1:]:
        coords_i = np.nonzero(distance_map == i)
        if len(coords_i) != 0:
            urns.append(np.stack(coords_i, axis=-1))

    # calculate the proba  depending on the number of different radius value
    if nb_val_rad == None:
        nb_urns = len(urns)
    else :
        nb_urns = min(len(urns), nb_val_rad)
    proba_urns = []
    proba_urns.append(0)
    for i in range(nb_urns):
        last = proba_urns[-1]
        prob_cum = last + (2 ** (nb_urns - (i + 1))) / ((2 ** nb_urns) - 1)
        proba_urns.append(prob_cum)

    # draw  the kind of vessel to be disconnected for each disconnection
    throw_urns = np.random.rand(nb_disconnection)

    disconnected_image = image.copy()

    #initialisation of the disconnections
    disconnections = np.zeros(image.shape)

    # creation of each mask of disconnection (chaque tache est differente)
    for i in range(len(proba_urns) - 1):
        # calculate the number of disconnections for each kind of disconnection
        category = (throw_urns > proba_urns[i]) * (throw_urns <= proba_urns[i + 1]) * 1
        number_throw_urn = np.sum(category)

        # draw the coordinates where disconnections are located
        point_disconnect = np.random.randint(len(urns[i]), size=number_throw_urn)

        #creation of each mask for  the i-th category of disconnection
        for j in point_disconnect:
            # parameters of the disconnection
            size_vessels = radius_vessels[i + 1]
            mean_size_disconnect = size_max_deco // size_vessels
            size_ball = abs(int(np.random.normal(mean_size_disconnect, scale=2)))
            if size_ball == 0:
                size_ball = 1

            # position of the center of the disconnection
            x = urns[i][j][0]
            y = urns[i][j][1]
            z = urns[i][j][2]

            # creation of the disconnection
            disconnect = create_simple_deconnexion(image, skelet, distance_map, x, y, z, size_ball)

            # disconnection on the source_2D
            disconnected_image = disconnected_image - disconnect
            disconnected_image = (disconnected_image == 1) * 1

            #mask of the disconnections
            disconnections = disconnections + disconnect
            disconnections = (disconnections >= 1) * 1

    return disconnected_image,  disconnections


def generator_noise(image, noise_size, threshold):
    """
   :param image: source_2D containing the vascular structure that we want add artefacts
   :param noise_size:frequency that represent noise ( low frequencies)
   :param threshold: threshold applied to obtain artefacts
    return the source_2D containing the vascular structure with artefacts
    """

    #size of the source_3D from which we add artefacts
    size = image.shape

    # creation of a uniform noise
    s = np.random.uniform(-1, 1, size)

    # Fourier transform
    S = np.fft.fftn(s)

    #creation  of a low pass filter
    high_frequencies = np.zeros(s.shape)
    filter = ball(noise_size)
    high_frequencies[0:noise_size + 1, 0:noise_size + 1, 0:noise_size + 1] = filter[noise_size:, noise_size:, noise_size:]

    # application of the filter
    S_filt = S * high_frequencies

    # inverse transform
    s_inv = np.fft.ifftn(S_filt)

    # normalisation to treshold
    s_inv = image_utils.normalize_image(s_inv, 1)

    # treshold and fusion of the source_2D and the artefact
    noise = (s_inv > threshold) * 1.0
    noise = image + noise
    noise = (noise > 0)

    # research of the bigger componant
    regions = label(noise, connectivity=2)
    counts = []
    counts.append(0)
    nb_composante_init = np.max(label(image, connectivity=2))

    #research of the bigger labels (containing vascular network with artefacts on it)
    for i in range(1, np.amax(regions) + 1):
        count = np.count_nonzero(regions == i)
        counts.append(count)

    #getting the connected composant of the initial vascular network
    counts_max = sorted(counts)[-nb_composante_init:]
    vascular_indices = []
    for i in counts_max:
        vascular_indices.append(counts.index(i))

    # substraction to delete the noise that are on the structures
    for vascular_indice in vascular_indices:
        noise = noise * 1.0 - (regions == vascular_indice) * 1.0

    # fusion
    noise = (noise > 0) * 1.0
    return noise


class BinaryDeconnect(MapTransform):
    """
   :param nb_disconnection: number of disconnection to add to vascular structure
   :param size_max_deco: mean maximal size of disconnection that can be applied
   :param nb_val_rad: number of centerline types that can be disconnected
   """

    def __init__(self, keys: KeysCollection,  nb_deco, size_max_deco = 8, nb_val_rad=None) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.nb_deco = nb_deco
        self.size_max_deco = size_max_deco
        self.nb_val_rad = nb_val_rad
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key],__ = create_disconnections(d[key], self.nb_deco, self.size_max_deco, self.nb_val_rad)
        return d

class AddArtefacts(MapTransform):
    """
   :param label: source_2D containing the vascular structure that we want add artefacts
   :param mean_artefacts: frequency that represent noise( low frequencies)
   :param threshold: threshold applied to obtain artefacts
    """

    def __init__(self, keys: KeysCollection, label, mean_artefacts = 20, threshold = 0.8) -> None:
        self.keys: Tuple[Hashable, ...] = ensure_tuple(keys)
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")
        self.label = label
        self.mean_artefacts = mean_artefacts
        self.threshold = threshold


    def __call__(self, data):
        d = dict(data)
        artefacts = generator_noise(d[self.label], self.mean_artefacts, self.threshold)
        for key in self.keys:
            d[key] = d[key] + artefacts
            d[key] = np.clip(d[key], 0, 1)
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


########################################################################################################################

def create_dataset(origin_directory, new_dataset_directory, nb_deco, size_max_deco, mean_artefacts, threshold):

    images = sorted(glob(f"{origin_directory}/binary_images/*"))
    device = "cpu"

    train_files = [{"label": img, "deco": img,"label_art": img} for img in images]
    print(train_files)
    trans = Compose(
        [
            LoadImaged(keys=["deco", "label", "label_art"]),
            ScaleIntensityd(keys=["deco", "label", "label_art"]),
            BinaryDeconnect(keys=["deco"], nb_deco=nb_deco, size_max_deco=size_max_deco),
            AddArtefacts(keys=["deco", "label_art"], label="label", mean_artefacts=mean_artefacts, threshold=threshold),
            AddChanneld(keys=["deco", "label", "label_art"]),
            ToTensord(keys=["deco", "label", "label_art"]),
        ]
    )
    check_ds = Dataset(data=train_files, transform=trans)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    for batch_data, i in zip(check_loader,  range(len(images))):
        print(f"---------------------------traitement de l'source_2D n°{i}-----------------------------")
        label, deco, label_art = batch_data["label"].to(device), batch_data["deco"].to(device), batch_data["label_art"].to(device)
        write_nifti(data=label.detach().cpu().squeeze().numpy(), file_name=f"{new_dataset_directory}/label_{i}.nii.gz", resample=False)
        write_nifti(data=deco.detach().cpu().squeeze().numpy(), file_name=f"{new_dataset_directory}/img_{i}.nii.gz", resample=False)
        write_nifti(data=label_art.detach().cpu().squeeze().numpy(), file_name=f"{new_dataset_directory}/seg_{i}.nii.gz", resample=False)
        pos_deco = label_art - deco
        non_art_deco = label - pos_deco
        write_nifti(data=non_art_deco.detach().cpu().squeeze().numpy(), file_name=f"{new_dataset_directory}/denoise_deconnected_{i}.nii.gz", resample=False)
        boule = ball(2)
        pos_deco = pos_deco.detach().cpu().squeeze().numpy()
        pos_deco = binary_dilation(pos_deco, boule)
        write_nifti(data=pos_deco, file_name=f"{new_dataset_directory}/pos_{i}.nii.gz", resample=False)