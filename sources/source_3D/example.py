import sys
from disconnect import create_dataset
from train import training
from post_treatement import post_treatement
from pretreatement import substract_background
from plug_and_play import reconnector_plug_and_play
sys.path.insert(0, "../../")
from sources.image_utils import save_nifti

# First step : creation of a dataset to train our reconnecting model
origin_directory = "../../volumes/binary_trees"
new_dataset_directory = "../../volumes/generated_dataset"
size_deco_max = 8
mean_artefacts = 30
nb_deco = 20
threshold = 0.8

create_dataset(origin_directory, new_dataset_directory, nb_deco, size_deco_max, mean_artefacts, threshold)

# Second step : once the dataset is created, we can train the model
name_dir_model = "../../modeles/3D_model_vascu"
type_training = "reconnect_denoise"
norm = "batch"
training(new_dataset_directory, name_dir_model, type_training, norm, max_epochs=10)

# last step : use it either as a post processing or as a regularisation term for a variational segmentation
#post-processing
segmentation_path = "../../volumes/segmentation_test/image_01.nii.gz"
result_path = "../../volumes/results/image_01_post_processed.nii.gz"
image_post_treated = post_treatement(segmentation_path, name_dir_model, 10)
save_nifti(image_post_treated,result_path, metadata_model=segmentation_path)


#plug and play
image_path = "../../volumes/curvilinear_structures_to_segment/origin/image_01.nii.gz"
mask_path = "../../volumes/curvilinear_structures_to_segment/mask/mask_01.nii.gz"
preprocessed_path = "../../volumes/curvilinear_structures_to_segment/preprocessed/01_preprocessed.nii.gz"
tv_weight = 0.008
kernel_radius = 15

#pretreatement in order to be able to use the chan-vese data fidelity term
image_preprocessed = substract_background(image_path, mask_path, kernel_radius)
save_nifti(image_preprocessed,preprocessed_path, metadata_model=image_path)

segment_8_bits_reco = reconnector_plug_and_play(preprocessed_path, tv_weight, name_dir_model, sigma=10e-3)






