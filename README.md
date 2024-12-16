# Learning a reconnecting model for curvilinear structures 

We proposed to train a **reconnecting model for curvilinear structures in 2D and in 3D** that can be used as: 
- a regularization term in an  unsupervised plug-and-play segmentation approach
- a post-processing step for any type of vascular segmentation result

This repo contains the code to apply methods presented in two different papers : 

[1] Sophie Carneiro-Esteves, Antoine Vacavant and Odyssée Merveille. "A plug-and-play framework for curvilinear structure segmentation based on a learned reconnecting regularization" Neurocomputing 2024

[2] Sophie Carneiro-Esteves, Antoine Vacavant and Odyssée Merveille. "Restoring Connectivity in Vascular Segmentations using a Learned Post-Processing Model" TGI3, MICCAI 2024 Workshop

This repository does not provide a single standalone script. Instead, **it offers a set of reusable functions designed to be integrated into your existing pipeline**. You can incorporate these functions at the appropriate stages of your workflow to customize and enhance your processing pipeline according to your specific needs.

## Code structure
The code is stocked in ```sources/```. 
- ```source_2D``` contains the python functions to process 2D images.
- ```source_3D``` contains the python functions to process 3D images.
- ```image_utils.py``` contains useful functions to open, normalize and save images (2D and 3D).

The files in ```source_2D``` and ```source_3D``` are:
- ```disconnect.py``` : contains the functions to disconnect a binary tree.
- ```train.py```: contains the functions to train a reconnecting model with a generated dataset.
- ```post_treatement.py```: contains the function to use the reconnecting model as a post processing on a curvilinear segmentation.
- ```pretreatement.py```: contains a function that delete the background of an image to have an image composed of two classes.
- ```plug_and_play.py``` :  contains the functions to apply an unsupervised plug-and-play segmentation approach using the reconnecting model.
- ```grad_div_interpolation.py```: contains the functions for gradient operations.
- ```example.py``` : is a script to run that apply the different proposed frameworks
- ```compared_methods.py```: contains different methods with which we compared ourselves with. 


## How to use it ?

To understand the order of operations, check out the ```example.py``` file. It demonstrates how to:

- Create a dataset.
- Train the neural network.
- Apply the trained model as a post-processing step.
- Use the trained model as a regularisation term for a plug-and-play segmentation pipeline.


If you don't have annotations or binary vascular trees to train a model, models used in [1] and [2] are available in the directory ```modeles```.




 
