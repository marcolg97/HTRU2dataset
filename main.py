# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""
import numpy as np
import plot
import load_data
import utils
import gaussianClassifier as gc
import logisticRegression as lr
import SVM
import GMM
import scoresRecalibration as sr
import experimentalResults as er
import ROC
import evaluationHyperparameter as eh

def ZNormalization(D, mean = None, standardDeviation = None):
    """
    It will scale the data but will not modify the distribution 
    (we will have the same distribution of the original data).
    This tranformed distribution has a mean of 0 and a standard deviation of 1

    Parameters
    ----------
    D : data matrix

    Returns
    -------
    normalizedData : normalized data with z-normalization

    """
    if (mean is None and standardDeviation is None):
        mean = D.mean(axis=1)
        standardDeviation = D.std(axis=1)
    # z = (x_i - mu) / sigma
    normalizedData = (D - utils.mcol(mean)) / utils.mcol(standardDeviation)
    return normalizedData, mean, standardDeviation



if __name__ == '__main__':
    
    # Load the training data
    D,L = load_data.load("data/Train.txt")
    
    
    #print("Data matrix: ", D)
    #print("labels", L)
    plot.plotFeatures(D, L, utils.featuresNames, utils.classesNames, "before")
    
    # We notice that the values are high, so to avoid possible problems with approximation later on
    # We normalize the data with z-normalization
    normalizedData, normalizedMean, normalizedStandardDeviation = ZNormalization(D)
    
    #print("Normalized data matrix: ", normalizedData)
    #print("labels", L)
    plot.plotFeatures(normalizedData, L, utils.featuresNames, utils.classesNames, "after")
    
    ''' ------------- CORRELATION ANALYSIS ------------- '''
    # We will plot the heatmap of the whole dataset and of the single class to analyze the correlation between features
    # Darker colors means higher correlation
    plot.heatmap(normalizedData, L, True)
    plot.heatmap(normalizedData, L, False)
    
    ''' ------------------ MVG CLASSIFIER ---------------- '''
    gc.computeMVGClassifier(normalizedData, L)


    ''' ----------------------------- LOGISTIC REGRESSION ----------------------------- '''
    
    lr.findBestLambda(normalizedData, L)
    
    # We select the best value of lambda from the DRAW
    lambd_lr = 1e-4
    
    lr.computeLogisticRegression(normalizedData, L, lambd = lambd_lr)
    
    
    ''' ----------------------------- SUPPORT VECTOR MACHINES -----------------------------'''
    
    ''' A. LINEAR SVM '''
    
    # # Find the right value of C for the unbalanced SVM from the draw
    SVM.findUnbalancedC(normalizedData, L, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))
   
    # # Find the right value of C for the balanced SVM with different priors from the draw
    SVM.findBalancedC(normalizedData, L, 0.5, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))
    SVM.findBalancedC(normalizedData, L, 0.1, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))
    SVM.findBalancedC(normalizedData, L, 0.9, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30))
    
    # # We select the best value of C from the DRAW (you can see it inside /SVM_balanced e /SVM_unbalanced)
    C_unbalanced = 10**-2      
    C_balanced_05 = 10**-3
    C_balanced_01 = 6*10**-3
    C_balanced_09 = 7*10**-4
    # # We find also the best value of K between 1.0 and 10.0
    K=1.0
    
    # # Compute min DCF for unbalanced SVM (no PCA , PCA=7, PCA=6) 
    SVM.computeLinearSVM(normalizedData, L, C=C_unbalanced, K=1.0, pi_T=None)
    
    # # Compute min DCF for balanced SVM (no PCA , PCA=7, PCA=6) for priors 0.5, 0.1 and 0.9
    SVM.computeLinearSVM(normalizedData, L, C_balanced_05, K=1.0, pi_T=0.5)
    SVM.computeLinearSVM(normalizedData, L, C_balanced_01, K=1.0, pi_T=0.1)
    SVM.computeLinearSVM(normalizedData, L, C_balanced_09, K=1.0, pi_T=0.9)
    
    ''' B. POLYNOMIAL SVM '''
    
    # # Find the right value of C and c from the draw
    SVM.findPolynomialC(normalizedData, L, rangeK=[0.0,1.0], d=2, rangec=[0,1,15], rangeC=np.logspace(-5, -1, num=30)) 
      
    # # We select the best value of C and c from the DRAW (you can see it inside /SVM_poly)
    C_polynomial = 5*10**(-5) 
    c_polynomial = 15
    
    # # Compute min DCF for POLYNOMIAL SVM (no PCA , PCA=7, PCA=6) for priors 0.5, 0.1 and 0.9
    SVM.computePolynomialSVM(normalizedData, L, C = C_polynomial, c = c_polynomial, K = 1.0, d = 2)
    
    '''B. RBF SVM'''
        
    # # Find the right value of C and gamma from the draw
    SVM.findRBFKernelC(normalizedData, L, rangeK = [0.0, 1.0], rangeC = np.logspace(-3, 3, num=30), rangeGamma = [10**(-4),10**(-3)])
    
    # # We select the best value of C and gamma from the DRAW (you can see it inside /SVM_RBF)
    C_RBF = 10**(-1)
    gamma_RBG = 10**(-3)
    
    # # Compute min DCF for RBF SVM (no PCA , PCA=7, PCA=6) for priors 0.5, 0.1 and 0.9
    SVM.computeRBFKernel(normalizedData, L, C = C_RBF, gamma = gamma_RBG, K = 1.0)
    
    ''' ----------------------------- GAUSSIAN MIXTURE MODEL ----------------------------- '''
    
    # # Find the right number of components for the full, diagonal and tied covariance
    GMM.findGMMComponents(normalizedData, L, maxComp = 7)
    
    # We find the best value of components for each model from the DRAW
    nComp_full = 4 # 2^4 = 16
    nComp_diag = 5 # 2^5 = 32
    nComp_tied = 6 # 2^6 = 64
    
    GMM.computeGMM(normalizedData, L, nComp_full, mode = "full")
    GMM.computeGMM(normalizedData, L, nComp_diag, mode = "diag") 
    GMM.computeGMM(normalizedData, L, nComp_tied, mode = "tied")  
    
    
    ''' ----------------------------- SCORES RECALIBRATION ----------------------------- '''
    
    sr.computeActualDCF(normalizedData, L, lambd = lambd_lr, components = nComp_full) 
    sr.computeBayesErrorPlots(normalizedData, L, lambd = lambd_lr, components = nComp_full)
    sr.calibratedBayesErrorPlots(normalizedData, L, lambd = lambd_lr, components = nComp_full)
    sr.computeCalibratedErrorPlot(normalizedData, L, lambd = lambd_lr, components = nComp_full)
    
    ''' ----------------------------- EXPERIMENTAL RESULTS ----------------------------- '''
        
    DT, LT = load_data.load("data/Test.txt")
    
    # On Z-normalization we use the mean and the standard deviation of the Z-normalization done on the training set
    normalizedDataTest, _, _ = ZNormalization(DT, normalizedMean, normalizedStandardDeviation)
    
    
    er.computeExperimentalResults(normalizedData, L, normalizedDataTest, LT)
    
    eh.EvaluateHyperParameterChosen(normalizedData, L, normalizedDataTest, LT)
    
    ROC.computeROC(normalizedData, L, normalizedDataTest, LT)
    
   