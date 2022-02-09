# -*- coding: utf-8 -*-
"""

@author: Giacomo Vitali and Marco La Gala

"""

import numpy as np
import utils
import folds
import cost_functions as cf
import dimensionality_reduction as dr

def trainGaussianClassifier(D, L):
    mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
    mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
       
    sigma0 = np.cov(D[:, L == 0])
    sigma1 = np.cov(D[:, L == 1])
    
    return mean0, sigma0, mean1, sigma1


def trainNaiveBayes(D, L):
    mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
    mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
              
    sigma0 = np.cov(D[:, L == 0])
    sigma1 = np.cov(D[:, L == 1])
    
    sigma0 = sigma0 * np.identity(sigma0.shape[0])
    sigma1 = sigma1 * np.identity(sigma1.shape[0])

    return mean0, sigma0, mean1, sigma1


def trainTiedCov(D, L):
    mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
    mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
       
    sigma0 = np.cov(D[:, L == 0])
    sigma1 = np.cov(D[:, L == 1])
    
    sigma = 1 / (D.shape[1]) * (D[:, L == 0].shape[1] * sigma0 + D[:, L == 1].shape[1] * sigma1)
    
    return mean0, sigma, mean1, sigma


def getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, evaluationSet):
        LS0 = logpdf_GAU_ND(evaluationSet, mean0, sigma0 )
        LS1 = logpdf_GAU_ND(evaluationSet, mean1, sigma1 )
        #log-likelihood ratios
        llr = LS1-LS0
        return llr

def logpdf_GAU_ND(x, mu, sigma):
    return -(x.shape[0]/2)*np.log(2*np.pi)-(0.5)*(np.linalg.slogdet(sigma)[1])- (0.5)*np.multiply((np.dot((x-mu).T, np.linalg.inv(sigma))).T,(x-mu)).sum(axis=0)


def computeMVGClassifier (D,L):
       
    # ------------- GAUSSIAN CLASSIFIER -------------
    # To understand which model is most promising, and to assess the effects of using PCA, we can adopt two methodologies: single-fold and k-fold
    # We decide to implement both of them. First, we implement the single-fold
    
    print("  Single-fold                K-fold")
    print("0.5  0.1  0.9            0.5  0.1  0.9")
    print("no PCA")
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(D, L, trainGaussianClassifier, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1)
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    # Then we implement the k-fold with k=5
    scoresKFold, evaluationLabelsKFold = folds.Kfold(D, L, trainGaussianClassifier, getScoresGaussianClassifier)
   
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
   
    
    print("Full Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- NAIVE BAYES -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(D, L, trainNaiveBayes, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(D, L, trainNaiveBayes, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Diagonal Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- TIED COVARIANCE -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(D, L, trainTiedCov, getScoresGaussianClassifier) 
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(D, L, trainTiedCov, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Tied Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ---------------------------------------- PCA m=7 ----------------------------------------
    
    print("PCA m = 7")
    PCA7 = dr.PCA(D, L, 7)
    
    # ------------- GAUSSIAN CLASSIFIER -------------
        
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA7, L, trainGaussianClassifier, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1)
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    # Then we implement the k-fold with k=3
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA7, L, trainGaussianClassifier, getScoresGaussianClassifier)
   
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
   
    
    print("Full Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- NAIVE BAYES -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA7, L, trainNaiveBayes, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA7, L, trainNaiveBayes, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Diagonal Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- TIED COVARIANCE -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA7, L, trainTiedCov, getScoresGaussianClassifier) 
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA7, L, trainTiedCov, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Tied Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ---------------------------------------- PCA m=6 ----------------------------------------
    
    print("PCA m = 6")
    PCA6 = dr.PCA(D, L, 6)
    
    # ------------- GAUSSIAN CLASSIFIER -------------
        
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA6, L, trainGaussianClassifier, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1)
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    # Then we implement the k-fold with k=3
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA6, L, trainGaussianClassifier, getScoresGaussianClassifier)
   
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
   
    
    print("Full Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- NAIVE BAYES -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA6, L, trainNaiveBayes, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA6, L, trainNaiveBayes, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Diagonal Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- TIED COVARIANCE -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA6, L, trainTiedCov, getScoresGaussianClassifier) 
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA6, L, trainTiedCov, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Tied Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ---------------------------------------- PCA m=5 ----------------------------------------
    
    print("PCA m = 5")
    PCA5 = dr.PCA(D, L, 5)
    
    # ------------- GAUSSIAN CLASSIFIER -------------
        
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA5, L, trainGaussianClassifier, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1)
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    # Then we implement the k-fold with k=3
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA5, L, trainGaussianClassifier, getScoresGaussianClassifier)
   
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
   
    
    print("Full Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- NAIVE BAYES -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA5, L, trainNaiveBayes, getScoresGaussianClassifier)   
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA5, L, trainNaiveBayes, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Diagonal Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    # ------------- TIED COVARIANCE -------------
    
    scoresSingleFold, evaluationLabelsSingleFold = folds.single_fold(PCA5, L, trainTiedCov, getScoresGaussianClassifier) 
    
    dcf_single_05 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.5,1,1) 
    dcf_single_01 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.1,1,1) 
    dcf_single_09 = cf.minimum_detection_costs(scoresSingleFold, evaluationLabelsSingleFold, 0.9,1,1) 
    
    
    scoresKFold, evaluationLabelsKFold = folds.Kfold(PCA5, L, trainTiedCov, getScoresGaussianClassifier)
    
    dcf_kfold_05 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.5,1,1) 
    dcf_kfold_01 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.1,1,1)
    dcf_kfold_09 = cf.minimum_detection_costs(scoresKFold, evaluationLabelsKFold, 0.9,1,1) 
    
    
    print("Tied Covariance")
    print("%.3f  %.3f  %.3f  %.3f  %.3f  %.3f" %(dcf_single_05, dcf_single_01, dcf_single_09, dcf_kfold_05, dcf_kfold_01, dcf_kfold_09))
    
    
    return
    