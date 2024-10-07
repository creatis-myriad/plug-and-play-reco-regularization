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

import numpy as np
import os
import sys
import time
from scipy.sparse import coo_matrix, diags, vstack
from math import floor


def unravel(ind_2d, dim):
    """
    Transforms 2d coordinates (x_1, x_2) into a 1d unique coordinate J
    ind_2d is a 2x1 vector of integer coordinates
    dim is a 2x1 vector of integer dimensions
    Order of dimensions is (y,x)
    """
    return ind_2d[1] + dim[1]*ind_2d[0]
    
def ravel(j, dim):
    """
    Transforms 1d unique vector coordinate into 2d
    Expected order of coordinates is y,x 
    """
    ind_2d = np.zeros(2, dtype=int)
    
    ind_2d[1] = j % dim[1]
    ind_2d[0] = (j / dim[1])%dim[0]
    return ind_2d

def gradient_2d_along_axis(dim, axis=0):
    """
    Computes the gradient operator given the dimensions 'dim' of the source_2D
    along the axis 'axis' Outputs a gradient matrix operator

    INPUT:
        - dim: np array ; dimensions of the source_2D ordered as [dimy, dimx]
        - axis: int, determine which gradient is computed
                    axis = 0: y gradient
                    axis = 1: x gradient
      
    OUTPUT: 
        - (A-I): coo_matrix ; gradient operator 
    """
    nb_pixels = dim[0] * dim[1]
    row = np.array(range(0, nb_pixels), dtype=int)
    col = np.zeros(nb_pixels, dtype=int) 
    Icol = np.arange(nb_pixels)
    data = np.ones(nb_pixels, dtype=int)
    canon = np.zeros(2, dtype=int)
    canon[axis] = 1

    for ind in range(nb_pixels):
        ind_2d = ravel(ind,dim)
        if (ind_2d[axis] == (dim[axis] - 1)): # border pixel ==> grad = 0
            col[ind] = ind
        else:
            col[ind] = unravel(ind_2d + canon, dim)

    A = coo_matrix((data,(row,col)), shape = (nb_pixels, nb_pixels))
    I = coo_matrix((data,(row,Icol)), shape = (nb_pixels, nb_pixels))

    return(A-I)

    
def gradient_2d_along_axis_anisotropy_correction(ori, axis=0):
    """
	Computes the gradient operator only for pixels which do not have an 
    orientation (ori[:,i,j].all() = 0) and which target pixel 
    (in the direction of the gradient) also do not have an orientation 
	Outputs a gradient matrix operator

	INPUT:
        - ori: np array ; orientations ordered such as ori = [oriy, orix]
        - axis: int, determine which gradient is computed
                    axis = 0: y gradient
                    axis = 1: x gradient
						
	OUTPUT: 
		- (A-I): coo_matrix ; gradient operator 
	"""

    dim = ori[0,:,:].shape
    nb_pixels = dim[1] * dim[0]

    row = np.arange(nb_pixels)
    col = np.arange(nb_pixels)
    Icol = np.arange(nb_pixels)
    data = np.ones(nb_pixels, dtype=int)
    
    canon = np.zeros(2, dtype = int)
    canon[axis] = 1 # canonical basis vector

    # pixels with no orientation
    mask_no_orientations = (np.sum(abs(ori), axis = 0) == 0).astype(int)
    pixels_no_orientations = np.nonzero(mask_no_orientations.ravel())[0]

    for ind in pixels_no_orientations:
        ind_2d = ravel(ind, dim) # current pixel 3d coordinates
        
        if (ind_2d[axis] == (dim[axis] - 1)): # border pixel ==> grad = 0
            col[ind] = ind
            
        else: 
            # orientation at target pixel (ind_3d + gradient direction)
            vec_ori = ori[:, ind_2d[0] + canon[0], ind_2d[1] + canon[1]] 
            
            # anisotropy correction:
            # compute the standard gradient only for pixels which 
            # target pixel do not have an orientation 
            if (vec_ori == 0).all():
                col[ind] = unravel(ind_2d + canon, dim)
  
    A = coo_matrix((data, (row, col)), shape = (nb_pixels, nb_pixels))
    I = coo_matrix((data, (row, Icol)), shape = (nb_pixels, nb_pixels))

    return(A-I)

def directional_2d_gradient(ori):
    """
    Computes the oriented gradient operator given the orientation source_2D 'ori'.
    The ori is an source_2D contains source_2D vectors that are either null
    (no orientation) or normalized (orientation).
    Outputs an oriented gradient matrix operator (and 0 at points without 
    orientation)

    The orientation source_2D is arranged in a source_3D matrix of shape (2, dimy, dimx)
    """
    
    dim = ori[0,:,:].shape
    nb_pixels = dim[0] * dim[1]

    # for the coefficient matrix
    row = []
    col = []
    data = []

    # for the identity matrix
    Irow = np.arange(nb_pixels) 
    Icol = np.arange(nb_pixels) 
    Idata = np.ones(nb_pixels) 

    # mask of the pixels with no orientation or which are border pixels
    mask_no_orientations = (np.sum(abs(ori), axis = 0) == 0).astype(int)
    mask_no_orientations[0,:] = 1
    mask_no_orientations[-1,:] = 1
    mask_no_orientations[:,0] = 1
    mask_no_orientations[:,-1] = 1

    pixels_no_orientations = np.nonzero(mask_no_orientations.ravel())[0]
    pixels_orientations = np.nonzero(mask_no_orientations.ravel()==0)[0]

    # If no orientation or border ==> add 1 which will be killed with the 
    # substraction of the identity matrix
    row.extend(pixels_no_orientations)
    col.extend(pixels_no_orientations)
    data.extend([1] * len(pixels_no_orientations))

    for ind in pixels_orientations:        
        ind_2d = ravel(ind, dim)
        Vori = ori[:, ind_2d[0],ind_2d[1]] ## orientation vector in order y,x

        # Compute the 9 coefficients for bilinear interpolation
        for dx2 in range(-1,2):
            for dx1 in range(-1,2):
                u = np.array([dx2,dx1], dtype=int)
                row.append(ind)
                col.append(unravel((ind_2d[0]+u[0], ind_2d[1]+u[1]), dim))
                mylambda = 1
                for d in range(0,2):
                    mylambda = mylambda * ((1. - abs(u[d]))*(1. - abs(Vori[d]))\
                    + abs(u[d]) * ( u[d] * Vori[d] + abs(Vori[d]) )/2.)
                data.append(mylambda)

    data = np.array(data)
    row = np.array(row)
    col = np.array(col)
    
    A = coo_matrix((data,(row,col)), shape=(nb_pixels,nb_pixels))
    I = coo_matrix((Idata,(Irow,Icol)), shape=(nb_pixels,nb_pixels))
   
    return(A-I)


def standard_gradient_operator_2d(op_grady, op_gradx):
    '''
        Construct the matrix formulation, V^{\nabla f}, of the 
        standard gradient \nabla f
    '''
    nb_pixels = op_gradx.get_shape()[0]

    row_x, col_x = op_gradx.nonzero()
    row_y, col_y = op_grady.nonzero()

    data_x = op_gradx.data
    data_y = op_grady.data

    row_x += nb_pixels

    all_rows = np.concatenate((row_y, row_x))
    all_cols = np.concatenate((col_y, col_x))
    all_data = np.concatenate((data_y, data_x))

    grad_matrix = coo_matrix((all_data, (all_rows, all_cols)), 
                              shape = (2 * nb_pixels, nb_pixels)
                            )
    return grad_matrix
    
def mixed_gradient_operator_2d(op_grady, op_gradx, op_grad_dir):
    '''
        Construct the matrix formulation, M^{\nabla m}, of the 
        mixed gradient \nabla m
    '''

    nb_pixels = op_gradx.get_shape()[0]
    row_x, col_x = op_gradx.nonzero()
    row_y, col_y = op_grady.nonzero()
    row_dir, col_dir = op_grad_dir.nonzero()

    data_x = op_gradx.data
    data_y = op_grady.data
    data_dir = op_grad_dir.data

    row_x += nb_pixels
    row_dir += 2 * nb_pixels

    all_rows = np.concatenate((row_y, row_x, row_dir))
    all_cols = np.concatenate((col_y, col_x, col_dir))
    all_data = np.concatenate((data_y, data_x, data_dir))

    grad_matrix = coo_matrix((all_data, (all_rows, all_cols)), 
                                shape = (3 * nb_pixels, nb_pixels)
                            )
    return grad_matrix
