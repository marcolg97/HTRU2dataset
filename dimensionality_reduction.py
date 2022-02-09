# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""


import numpy as np
import utils

def PCA(D, L, m):
    
    # 1° calcolate covariance matrix
    C = covariance(D, L)
    
    # 2° calcolate eigenvectors eigenvalues
    
    # Method A: for generic square matrix use numpy.linalg.eig
    #s, U = EIG_eigenvectors_eigenvalues(C, D, L, m)       

    # Method B: for symmetric matrix use numpy.linalg.eigh
    s, U = EIGH_eigenvectors_eigenvalues(C, D, L, m)

    # Method C: matrix semi-definite positive we can use SVD (single 
    # value decomposition) to get sorted eigenvectors and eigenvalues 
    #s, U = computePCAsecondVersion(C,D,L, m) 
        
    # Principal components
    P = U[:, 0:m]
    # We can apply the projection to the matrix
    # PCA projection matrix
    DP = np.dot(P.T, D)
    
    return DP

def covariance(D, L):
    
    """ Principal Component Analysis
    C = 1 / N * sommatoria_1aN( (x_i - mu) * (x_i - mu)^T )
    N is the number of sample, so the column of the matrix
    
    
    mu is the dataset mean mu=1/N*sommatoria_1aN(x_i)
    
    """
    
    #1° Find the mean of the matrix
    
    """
    # This method works but it is not permormance because for loop are slow in pyton
        mu = 0
        for i in range(D.shape[1]):
            mu = mu + D[:, i:i+1]
        
        mu = mu / float(D.shape[1])
    """
    # So we use the numpy method called mean
    
    # Compute mean over columns of the dataset matrix (mean over columns means that
    # for the first row we get a value, for the second row we get a value, ecc.)
    mu = D.mean(axis=1)
    
    #Better way is to use the mean method of numpy
    #We need a nx1 matrix but we obtain the 1xn (1-D), so we reshape it through 
    #the vcol method define above
    
    # We want to subtract mean to all elements of the dataset nxm (with broadcasting)
    # We need to reshape the 1-D array mu to a column vector nx1
    # 3x4   - 1x3 IS NOT POSSIBLE --> we need to do 3X4 - 3X1
    #       flower1 flower2 flower3 flower4
    # att1      *       *       *       *
    # att2      *       *       *       *
    # att3      *       *       *       *
    # We subtruct each column with the mean (normalize)
    #       flower1 flower2 flower3 flower4
    # att1      *-mu    *-mu    *-mu    *-mu
    # att2      *-mu    *-mu    *-mu    *-mu
    # att3      *-mu    *-mu    *-mu    *-mu
   
    # So we reshape the 1xn to nx1
    mu = utils.mcol(mu)
    
    # Now we can subtract (with broadcasting)
    #We center the matrix removing the mean
    # DC =  Matrix of centered data
    DC = D - mu
    
    # 2 - Covariance matrix
    
    """
        Now I can computing the data covariance matrix
        C = 0
        for i in range(D.shape[1]):
            
            C = C + numpy.dot(D[:, i:i+1] - mu, (D[:, i:i+1] - mu).T)
        
        C = C / float(D.shape[1])
        
        # Attention: D[:, 1:2] and NOT D[:, 1] becuause the first is a 
        # MATRIX COLUMN, the second i a vector (1xn)
        ------------------
        [[4.9]
         [3. ]
         [1.4]
         [0.2]]
        ------------------
        [4.9    3.  1.4     0.2]
        ------------------
        
        
        print(D)
        
        print("------------------")
        
        print(D[:, 1:2]-mu)
        
        print("------------------")
        
        print((D[:, 1:2]-mu).T)
        
        print(numpy.dot(D[:, 1:1+1] - mu, (D[:, 1:1+1] - mu).T))
    """      
    
    
    
    # Now we can compute the covariance matrix
    # DC.shape[1] is 150, it's the N parameter
    C = (1/DC.shape[1]) * (np.dot(DC, DC.T))
    
    return C

'''
def EIG_eigenvectors_eigenvalues(C, D, L, m):
    # with eig the eigenvalues and eigenvectors are not sorted 
    s, U = np.linalg.eig(C)
    
    # For this reason, to sort them, I use argsort, 
    # sorting them from the largest to the smallest
    idx = s.argsort()[::-1]
    s1 = s[idx]
    U1 = U[:,idx]
        
    
    return s1, U1
'''

def EIGH_eigenvectors_eigenvalues(C, D, L, m):
    # Get the eigenvalues (s) from smaller to largest and eigenvectors (columns of U) of C
    s, U = np.linalg.eigh(C)
    
    # We make in descending order 
    U = U[:, ::-1] 
    
    return s, U