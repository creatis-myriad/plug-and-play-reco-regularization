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
    
def proj(image):
    
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def prox_TV_FGP_matrix_form(image, grad_matrix, lipschitz, 
                                 lambdA, epsilon, max_iter = 500):
    
    '''
    Compute the prox of TV in the constrained case by the FGP algorithm [1]. 
    The constraint is done by a projection on a convex set, which is done here 
    by the "proj" function (projection in the set [0,1])

    INPUT: 
        - image: 2d or 3d numpy array; Input image
        - grad_matrix: 2d np array; Incidence matrix
        - lipschitz: int; constante de lipschitz, depends on the number of edges 
                    (8 for 2 edges, 16 for 4 edges per pixel)
        - lambdA: float; Regularization weight balancing between Edata and Ereg
                
        - epsilon: float; Convergence criterion of the prox computation. 
                            If energy < epsilon, the prox algorithm stops.
        - max_iter: int; Maximum iteration number of the FGP algorithm
        
    OUTPUT:
        - res: 2d np array; the prox of TV constrained
        - nbIter: int; number of iterations computed by the FGP
        - energy_vec: float; final energy at convergence
        
    REFERENCES:
        [1] A. Beck and M. Teboulle, "Fast gradient-based algorithms for 
        constrained total variation image denoising and deblurring problems", 
        Trans. Image Process. 2009
    '''
   
    grad_matrix_t = grad_matrix.getH()
    image_flatten = image.flatten()

    dimy, __ = grad_matrix.shape
    
    rs = np.zeros(dimy, np.float)
    pq = np.zeros(dimy, np.float)
    
    res = np.zeros_like(image_flatten, np.float)
    
    energy = 100
    t = 1.
    nb_iter = 0
    
    energy_vec = []
    while (energy > epsilon and nb_iter < max_iter) :
        # Divergence
        div = grad_matrix_t.dot(rs)
              
        out = proj(image_flatten - lambdA * div)        
        grad = grad_matrix.dot(out)
    
        pq_new = grad
        pq_new *= 1./ (lipschitz * lambdA)

        rs += pq_new

        # The projection on the unit ball is equivalent to a l2 norm
        norm = np.sqrt(((grad_matrix_t < 0).astype(int)).dot(rs * rs))
        norm = (((grad_matrix < 0).astype(int)).dot(norm))
        pq_new = rs / np.maximum(np.ones(norm.shape), norm)

        t_new = 1. / 2 * (1 + np.sqrt(1 + 4 * t**2))

        rs = pq_new + ((t - 1) / t_new) * (pq_new - pq)
        
        pq = pq_new
        t = t_new

        # Energy
        energy = np.linalg.norm(out - res)
        energy_vec.append(energy)
        res = out
        nb_iter += 1

    res = res.reshape(image.shape)
    print("nb prox iter:", nb_iter, "; energy:" , energy)

    return res, nb_iter, energy_vec
    
