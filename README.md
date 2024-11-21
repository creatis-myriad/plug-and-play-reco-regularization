# Learning a reconnecting model for curvilinear structures 

We proposed to train a **reconnecting model for curvilinear structures in 2D and in 3D** that can be used as: 
- a regularization term in an  unsupervised plug-and-play segmentation approach
- a post-processing step for any type of vascular segmentation result

This repo contains the code to apply methods presented in two different papers : 
- Sophie Carneiro-Esteves, Antoine Vacavant and Odyssée Merveille. "A plug-and-play framework for curvilinear structure segmentation based on a learned reconnecting regularization" Neurocomputing 2024
- Sophie Carneiro-Esteves, Antoine Vacavant and Odyssée Merveille. "Restoring Connectivity in Vascular Segmentations using a Learned Post-Processing Model" TGI3, MICCAI 2024 Workshop


## Code structure
The code is stocked in ```sources/```. 
- ```source_2D``` contains the python functions that permits to treat 2D images.
- ```source_3D``` contains the python functions that permits to treat 3D images.
- ```image_utils.py``` contains useful functions to open, normalize and save images (2D and 3D).
- ```metrics.py``` contains the different metrics that have been used to evaluate our methods.

The files in ```source_2D``` and ```source_3D``` are:
- ```disconnect.py``` : contains the functions to disconnect a binary tree.
- ```train.py```: contains the functions to train a reconnecting model with a generated dataset.
- ```post_treatement.py```: contains the function to use the reconnecting model as a post processing on a curvilinear segmentation.
- ```pretreatement.py```: contains a function that delete the background of an image to have an image composed of two classes.
- ```plug_and_play.py``` :  contains the functions to apply an unsupervised plug-and-play segmentation approach using the reconnecting model.
- ```grad_div_interpolation.py```: contains the functions for gradient operations.
- ```example.py``` : is a script to run that apply the different proposed frameworks
- ```compared_methods.py```: contains different methods with which we compared ourselves with. 



 
