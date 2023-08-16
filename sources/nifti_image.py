import numpy as np
import nibabel as nib


def read_nifti(path):
    img = nib.load(path)

    return np.array(img.dataobj)


def save_nifti(array, path, metadata_model=None):
    """
    Save a nd array to a nifti image on the disk
    Parameters
    ----------
    array : nd array 
        the image to save
    path : string
        location to save the image
    metadata_model : string, optional
        the path to a nifti image.
        The image will be saved with the same header as metadata_model. Otherwise, a default header will be provided

    Returns
    -------
    None.

    """
    if metadata_model is None:
        res = nib.Nifti1Image(array, affine=np.eye(4))
    else:
        nib_image = nib.load(metadata_model)
        res = nib.Nifti1Image(array, affine=nib_image.affine, header=nib_image.header)
        
    nib.save(res, path)
