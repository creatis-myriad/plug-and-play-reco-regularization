import os
import sys

sys.path.insert(0,"../sources")

from sources import image_utils

def compute_RORPO_2D():
    """
    Compute rorpo on the DRIVE dataset.

    """
    min_path = 10
    factor = 2
    nbScales = 5
    robustParam = 0
    files_to_treat = "image_optimization/image_background_substraction"
    directory_to_save = "image_optimization/rorpo"



    patient_list = ["%.2d" % i for i in range(1,41)]
    for patient in patient_list :
        print(f"****************** patient n° {patient} ******************** ")
        image_path = f"{files_to_treat}/image_{patient}_bg_substract_15.png"
        image_to_save = f"{directory_to_save}/RORPO_{patient}_{min_path}_{factor}_{nbScales}_{robustParam}.png"
        image_to_save_vx = f"{directory_to_save}/RORPO_{patient}_{min_path}_{factor}_{nbScales}_{robustParam}_vx.png"
        image_to_save_vy = f"{directory_to_save}/RORPO_{patient}_{min_path}_{factor}_{nbScales}_{robustParam}_vy.png"

        #to adapt in function of your installation of RORPO2D
        os.system(f"/home/carneiro/opt/RORPO2D/build/RORPO2D {image_path} {min_path} {factor} {nbScales} {robustParam} {image_to_save}")

        image_to_save_dirx = f"{directory_to_save}/RORPO_{patient}_{min_path}_{factor}_{nbScales}_{robustParam}_dirx.tif"
        image_to_save_diry = f"{directory_to_save}/RORPO_{patient}_{min_path}_{factor}_{nbScales}_{robustParam}_diry.tif"

        vx = image_utils.read_image(image_to_save_vx)/100 - 1
        vy = image_utils.read_image(image_to_save_vy)/100 - 1
        image_utils.save_image(vx, image_to_save_dirx)
        image_utils.save_image(vy, image_to_save_diry)


compute_RORPO_2D()