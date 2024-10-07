import sys

from disconnect import create_dataset
from train import training
from post_treatement import post_treatement
from pretreatement import substract_background
from plug_and_play import reconnector_plug_and_play
sys.path.insert(0, "../../")
from sources.image_utils import show_image, save_image

# First step : creation of a dataset to train our reconnecting model
origin_directory = "../../images/binary_trees"
new_dataset_directory = "../../images/generated_dataset"
size_deco_max = 8
noise_level = 200
nb_deco = 100

create_dataset(origin_directory, new_dataset_directory, nb_deco, size_deco_max, noise_level)


# Second step : once the dataset is created, we can train the model
name_dir_model = "../../modeles/2D_model_cco"
type_training = "reconnect_denoise"
norm = "batch"
training(new_dataset_directory, name_dir_model, type_training, norm, max_epochs=10)


# last step : use it either as a post processing or as a regularisation term for a variational segmentation
#post-processing
segmentation_path = "../../images/segmentation_test/image_01.png"
image_post_treated = post_treatement(segmentation_path, name_dir_model, 10)
show_image(image_post_treated, 'post-processed segmentation')


#plug and play
image_path = "../../images/curvilinear_structures_to_segment/origin/01_test.tif"
mask_path = "../../images/curvilinear_structures_to_segment/mask/01_test_mask.gif"
preprocessed_path = "../../images/curvilinear_structures_to_segment/preprocessed/01_preprocessed.png"
tv_weight = 0.008
kernel_radius = 15

#pretreatement in order to be able to use the chan-vese data fidelity term
image_preprocessed = substract_background(image_path, mask_path, kernel_radius)
save_image(image_preprocessed,preprocessed_path)

segment_8_bits_reco = reconnector_plug_and_play(preprocessed_path, tv_weight, name_dir_model, sigma=10e-3)
show_image(segment_8_bits_reco, 'plug-and-play segmentation')






