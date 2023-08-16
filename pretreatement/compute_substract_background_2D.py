import sys
from skimage.color import rgb2grey
from skimage.filters import median
from skimage.morphology import disk
from skimage.morphology import binary_erosion
sys.path.insert(0,"../../sources")
from sources import image_utils
import numpy as np
import skimage
image_path = "/home/carneiro/Documents/Master/myPrimalDual/cracks/origin/Volker_DSC01663_690_143_973_1038.jpg"

# image_path = "/home/carneiro/Documents/Master/myPrimalDual/cracks/origin/Volker_DSC01663_690_143_973_1038.jpg"
# image_path = "/home/carneiro/Documents/Master/myPrimalDual/cracks/origin/brain_alexandre.png"
# image_path = "/home/carneiro/Documents/Master/myPrimalDual/cracks/origin/632685_sat.jpg"
# image_path = "/home/carneiro/Documents/Master/myPrimalDual/cracks/origin/angio_gray.png"
# image_path = "/home/carneiro/Documents/Master/myPrimalDual/cracks/origin/angio_gray.png"
# image_path = "/home/carneiro/Documents/Master/myPrimalDual/cracks/origin/ARBRE12.png"


kernel_radius = 30
image = image_utils.read_image(image_path)

#Convert the image to grey level
# image = image_utils.normalize_image(image_utils.read_image(image_path)[:,:,0])

grey_image = image_utils.normalize_image(rgb2grey(image))
# image_utils.save_image(((grey_image)*255).astype(np.uint8), 'test_grey.png')
# image_utils.save_image(((image_R)*255).astype(np.uint8), 'test_grey_R.png')
print(image.shape)#Invert the image to obtain white blood vessels
# grey_image = np.amax(grey_image) - (grey_image)
# grey_image = np.resize([512, 512])
# grey_image = skimage.transform.resize(grey_image, [512, 512])
grey_int16 = (grey_image * 255).astype(np.int16)

# #Substract the background
bg_substract = grey_int16 - median(grey_int16, disk(kernel_radius))
bg_substract[bg_substract<0] = 0

# # Apply the FOV mask
result = bg_substract

#Extend the dynamic between 0 and 255
result = image_utils.normalize_image(result) * 255

# Save the result
# output_path = f"cracks/pretreated/632685_sat_{kernel_radius}.png"
output_path = f"cracks/pretreated/crack_grey.png"
image_utils.save_image(grey_int16.astype(np.uint8), output_path)
