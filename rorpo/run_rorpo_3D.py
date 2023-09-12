import pandas as pd
import numpy as np
from sources import nifti_image
from sources.rorpo_3d import compute_rorpo_multiscale
from monai.transforms import Orientation
from monai.data import write_nifti
from sources import image_utils
import nibabel as ni




def compute_RORPO_3D():
    """
    Compute rorpo on the IRCAD volumes based on the results proposed by the work of Lamy et al.

    """

    parameters_to_use = pd.read_csv("Ircad_iso_SS_RORPO_Organ_Best_MCC_per_volume_summary.csv")
    dilat_size = 0

    # number of core to adapt in function of your machine
    nb_core = 4


    volumes = []
    lengths = []
    f_scale = []
    n_scale = []

    for i in range(0, 20):
        volumes.append(parameters_to_use.loc[i, "SerieName"].split('/')[1])
        lengths.append(int(parameters_to_use.loc[i, "VolumeName"].split('-')[0]))
        f_scale.append(float(parameters_to_use.loc[i, "VolumeName"].split('-')[1]))
        n_scale.append(int(parameters_to_use.loc[i, "VolumeName"].split('-')[2][0]))

    for name, min_path, factor, nbScales in zip(volumes, lengths, f_scale, n_scale):
        print(f"********************** {name} **********************")
        image_path = f"../../datas/ircad_iso_V3/{name}/patientIso.nii"
        mask_path = f"../../datas/ircad_iso_V3/{name}/maskedLiverAndVesselsIso.nii"
        image_to_save = f"../../datas/ircad_iso_V3/pretreated_ircad_10/{name}/rorpo_{min_path}_{factor}_{nbScales}_{dilat_size}"
        image = nifti_image.read_nifti(image_path)
        mask = image_utils.normalize_image(nifti_image.read_nifti(mask_path)).astype(np.uint8)
        scaling = ni.load(image_path).header["pixdim"][1]
        min_image = np.min(image)
        image = image + abs(min_image)
        min_path = int(min_path / scaling)

        multiscale_rorpo, vx_multiscale, vy_multiscale, vz_multiscale = compute_rorpo_multiscale(image, min_path, factor, nbScales, dilat_size=dilat_size,
                                 core=nb_core)
        multiscale_rorpo = multiscale_rorpo * mask

        #reorientation des volumes car je suis un boloss
        orient = Orientation(axcodes="RPS")
        orient2 = Orientation(axcodes="RAI")

        multiscale_rorpo = np.expand_dims(multiscale_rorpo, axis=0)
        vx_multiscale = np.expand_dims(vx_multiscale, axis=0)
        vy_multiscale = np.expand_dims(vy_multiscale, axis=0)
        vz_multiscale = np.expand_dims(vz_multiscale, axis=0)

        multiscale_rorpo = orient(multiscale_rorpo)[0]
        vx_multiscale = orient(vx_multiscale)[0]
        vy_multiscale = orient(vy_multiscale)[0]
        vz_multiscale = orient(vz_multiscale)[0]


        multiscale_rorpo = np.expand_dims(multiscale_rorpo, axis=0)
        vx_multiscale = np.expand_dims(vx_multiscale, axis=0)
        vy_multiscale = np.expand_dims(vy_multiscale, axis=0)
        vz_multiscale = np.expand_dims(vz_multiscale, axis=0)


        multiscale_rorpo = orient2(multiscale_rorpo)[0]
        vx_multiscale = orient2(vx_multiscale)[0]
        vy_multiscale = orient2(vy_multiscale)[0]
        vz_multiscale = orient2(vz_multiscale)[0]

        write_nifti(data=multiscale_rorpo,
                    file_name=f"{image_to_save}_intensity.nii.gz", resample=False)
        write_nifti(data=vx_multiscale,
                    file_name=f"{image_to_save}_dirx.nii.gz", resample=False)
        write_nifti(data=vy_multiscale,
                    file_name=f"{image_to_save}_diry.nii.gz", resample=False)
        write_nifti(data=vz_multiscale,
                    file_name=f"{image_to_save}_dirz.nii.gz", resample=False)

compute_RORPO_3D()