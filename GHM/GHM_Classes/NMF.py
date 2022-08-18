# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:16:11 2022

@author: Bryan
"""

import numpy as np
import warnings
from tqdm import trange


warnings.filterwarnings("ignore")
class NMF:
    def __init__(self):
        pass
    
    def coding(self, X, W, H0, 
              r=None, 
              a1=0, #L1 regularizer
              a2=0, #L2 regularizer
              sub_iter=[5], 
              stopping_grad_ratio=0.0001, 
              nonnegativity=True,
              subsample_ratio=1):
        """
        Find \hat{H} = argmin_H ( || X - WH||_{F}^2 + a1*|H| + a2*|H|_{F}^{2} ) within radius r from H0
        Use row-wise projected gradient descent
        """
        H1 = H0.copy()
        i = 0
        dist = 1
        idx = np.arange(X.shape[1])
        if subsample_ratio>1:  # subsample columns of X and solve reduced problem (like in SGD)
            idx = np.random.randint(X.shape[1], size=X.shape[1]//subsample_ratio)
        A = W.T @ W ## Needed for gradient computation
        grad = W.T @ (W @ H0 - X)
        while (i < np.random.choice(sub_iter)):
            step_size = (1 / (((i + 1) ** (1)) * (np.trace(A) + 1)))
            H1 -= step_size * grad 
            if nonnegativity:
                H1 = np.maximum(H1, 0)  # nonnegativity constraint
            i = i + 1
        return H1
    
    def ALS(self, X,
            n_components = 10, # number of columns in the dictionary matrix W
            n_iter=100,
            a0 = 0, # L1 regularizer for H
            a1 = 0, # L1 regularizer for W
            a12 = 0, # L2 regularizer for W
            H_nonnegativity=True,
            W_nonnegativity=True,
            compute_recons_error=False,
            subsample_ratio = 10):
        
            '''
            Given data matrix X, use alternating least squares to find factors W,H so that 
                                    || X - WH ||_{F}^2 + a0*|H|_{1} + a1*|W|_{1} + a12 * |W|_{F}^{2}
            is minimized (at least locally)
            '''
            
            d, n = X.shape
            r = n_components
            
            #normalization = np.linalg.norm(X.reshape(-1,1),1)/np.product(X.shape) # avg entry of X
            #print('!!! avg entry of X', normalization)
            #X = X/normalization

            # Initialize factors 
            W = np.random.rand(d,r)
            H = np.random.rand(r,n) 
            # H = H * np.linalg.norm(X) / np.linalg.norm(H)
            for i in trange(n_iter):
                #H = coding_within_radius(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
                #W = coding_within_radius(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
                H = self.coding(X, W.copy(), H.copy(), a1=a0, nonnegativity=H_nonnegativity, subsample_ratio=subsample_ratio)
                W = self.coding(X.T, H.copy().T, W.copy().T, a1=a1, a2=a12, nonnegativity=W_nonnegativity, subsample_ratio=subsample_ratio).T
                W /= np.linalg.norm(W)
                if compute_recons_error and (i % 10 == 0) :
                    print('iteration %i, reconstruction error %f' % (i, np.linalg.norm(X-W@H)**2))
            return W, H
