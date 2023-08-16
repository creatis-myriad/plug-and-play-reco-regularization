import numpy as np
import os
import argparse
from glob import glob
import nibabel as ni

from monai.data import write_nifti

def cropping(image, size_patch, overlap):
    crops = []
    shape_ = image.shape
    step = int(size_patch * overlap)
    for i in range(int(shape_[0] / step)):
        for j in range(int(shape_[1] / step)):
            for k in range(int(shape_[2] / step)):
                #traitements des patches sur les limites
                if i * step + size_patch > shape_[0]:
                    x_min = shape_[0] - size_patch
                    x_max = shape_[0]
                else:
                    x_min = i * step
                    x_max = i * step + size_patch

                if j * step + size_patch > shape_[1]:
                    y_min = shape_[1] - size_patch
                    y_max = shape_[1]
                else:
                    y_min = j* step
                    y_max =  j * step + size_patch

                if k * step + size_patch > shape_[2]:
                    z_min = shape_[2] - size_patch
                    z_max = shape_[2]
                else:
                    z_min = k * step
                    z_max = k * step + size_patch

                crop = image[x_min : x_max, y_min:y_max, z_min : z_max]
                crops.append(crop)
    return crops


parser = argparse.ArgumentParser(description='parametre pour la creation de patchs')
parser.add_argument('dataset', type=str, default="bullit", help='dataset selectionné')
parser.add_argument('taille', type=int, default=96, help='dataset selectionné')
parser.add_argument('overlap', type=float, default=0.5, help='dataset selectionné')
args = parser.parse_args()


size_patch = args.taille
overlap = args.overlap

if args.dataset == "vascu":
    path_images = "new_vascusynth_deco_rad_4"
elif args.dataset =="Brava":
    path_images = "brava_deco"
elif args.dataset =="Brava_2":
    path_images = "brava_deco_2"
else:
    exit()
name_images = sorted(glob(f"{path_images}/*nii.gz"))
new_directory = f"{path_images}/patches_{size_patch}_{overlap}"

if f"patches_{size_patch}_{overlap}" not in os.listdir(path_images):
    os.mkdir(new_directory)
else:
    print("job deja fait... veux tu vraiment le refaire ??? espece de boloss va")
    exit()

for path in name_images:
    image = ni.load(path).get_fdata()
    crops = cropping(image, size_patch, overlap)
    name = path.split("/")[-1].split(".")[0]
    print(f"traitement de l'image {name}")
    os.mkdir(f"{new_directory}/{name}")
    for i, crop in enumerate(crops):
        new_path =f"{new_directory}/{name}/{name}_{i}.nii.gz"
        write_nifti(data=crop, file_name= new_path, resample=False)
