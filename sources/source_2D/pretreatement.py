"""
Copyright or © or Copr. Odyssee Merveille (2019)
odyssee.merveille@gmail.com

This software is a computer program whose purpose is to reproduce the results
of the article "nD variational restoration of curvilinear structures with
prior-based directional regularization", O. Merveille, B. Naegel, H. Talbot
and N. Passat, IEEE Transactions on Image Processing, 2019
https://hal.archives-ouvertes.fr/hal-01832636.

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""
import sys
from skimage.color import rgb2gray
from skimage.filters import median
from skimage.morphology import disk
from skimage.morphology import binary_erosion
sys.path.insert(0,"../../sources")
from sources import image_utils
import numpy as np

def substract_background(image_path, mask_path, kernel_radius):
    image = image_utils.read_image(image_path)
    mask = image_utils.read_image(mask_path) * 1.0
    # Convert the image to grey level
    grey_image = image_utils.normalize_image(rgb2gray(image))
    mask = image_utils.normalize_image(mask)
    print(grey_image.shape, mask.shape)
    # Invert the image to obtain white blood vessels
    grey_image = np.amax(grey_image) - grey_image
    grey_int16 = (grey_image * 255).astype(np.int16)

    # Substract the background
    bg_substract = grey_int16 - median(grey_int16, disk(kernel_radius))
    bg_substract[bg_substract < 0] = 0

    # Erod the FOV to remove remaining white borders
    fov_dilat = binary_erosion(mask, disk(4)).astype(np.uint8)

    # Apply the FOV mask
    result = bg_substract * fov_dilat

    # Extend the dynamic between 0 and 255
    result = image_utils.normalize_image(result, 1) * 255
    result = result.astype(np.uint8)
    return result
