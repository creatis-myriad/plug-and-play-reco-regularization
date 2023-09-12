import numpy as np
from monai.data import Dataset, DataLoader, write_nifti
from skimage import morphology, filters
from glob import glob
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ToTensord,
    ScaleIntensityd,
    MaskIntensityd,
    AddChanneld,
)

# ircad files
#toutes les images
median_size = 10

img_file = "../../datas/ircad_iso_V3"
images = sorted(glob(os.path.join(img_file, "3Dircadb*/maskedLiverIso.nii")))
gts = sorted(glob(os.path.join(img_file, "3Dircadb*/vesselsIso.nii")))
masks = sorted(glob(os.path.join(img_file, "3Dircadb*/liverMaskIso.nii")))
file_pretreated_results = sorted(os.listdir("../../datas/ircad_iso_V3/pretreated_ircad_"+ str(median_size)))
files = [{"image": img, "label": gt, "mask" : mask} for img, gt, mask in zip(images, gts,masks)]

device = "cpu"


transforms = Compose(
    [
        LoadImaged(keys=["image", "label", "mask"]),
        Orientationd(keys=["image", "label", "mask"], axcodes="RAS"),
        ScaleIntensityd(keys=["image", "label", "mask"]),
        AddChanneld(keys=["image", "label", "mask"]),
        MaskIntensityd(keys=["image", "label"],mask_data= None, mask_key= "mask"),
        ToTensord(keys=["image", "label", "mask"]),
    ]
)

check_ds = Dataset(data=files, transform=transforms)
check_loader = DataLoader(check_ds, batch_size=1, num_workers=1, shuffle=False)

i =0
for batch_data in check_loader:
    inputs, labels, masks = (batch_data["image"].to(device), batch_data["label"].to(device), batch_data["mask"].to(device))
    inputs = inputs.squeeze().numpy()
    labels = labels.squeeze().numpy()
    masks = masks.squeeze().numpy()

    inputs = (inputs*255).astype(np.uint8)

    ball = morphology.ball(median_size)
    background = filters.median(inputs, ball)
    background = inputs.astype(np.int16) - background
    background[background < 0] = 0
    masks = morphology.binary_erosion(masks, morphology.ball(4))

    image_pre_processed = background * masks
    write_nifti(data = inputs, file_name ="../../datas/ircad_iso_V3/pretreated_ircad_"+ str(median_size)+"/" + file_pretreated_results[i] + "/inputs.nii.gz", resample = False)
    write_nifti(data = masks, file_name ="../../datas/ircad_iso_V3/pretreated_ircad_"+ str(median_size)+"/" + file_pretreated_results[i] + "/masks.nii.gz", resample = False)
    write_nifti(data = image_pre_processed, file_name ="../../datas/ircad_iso_V3/pretreated_ircad_"+ str(median_size)+"/" + file_pretreated_results[i] + "/preprocessed.nii.gz", resample = False)
    write_nifti(data = labels, file_name ="../../datas/ircad_iso_V3/pretreated_ircad_"+ str(median_size)+"/" + file_pretreated_results[i] +"/labels.nii.gz", resample = False)
    i = i + 1


