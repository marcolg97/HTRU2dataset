# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""

import numpy as np
import cost_functions as cf
import dimensionality_reduction as dr
import gaussianClassifier as gc
import scoresRecalibration as sr
import logisticRegression as lr
import GMM
import plot

def computeTiedCov(D, L, Dtest, Ltest, lambd, prior = 0.5):
    
    print("Tied-Cov PCA m = 7")
    
    TPR = []
    FPR = []
    mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(D, L)
    
    tiedcov_scores = gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest)
    scores = sr.calibrateScores(tiedcov_scores, Ltest, lambd).flatten()
    sortedScores = np.sort(scores)
    
    for t in sortedScores:
        confusionMatrix = cf.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(scores, Ltest, t)
        FPRtemp, TPRtemp = cf.computeFPRTPR(prior, 1, 1, confusionMatrix)
        TPR.append(TPRtemp)
        FPR.append(FPRtemp)
        
    return (TPR, FPR)

def computeLR(D, L, Dtest, Ltest, lambd, prior = 0.5):
    
    print("LR PCA m = 7")
    
    TPR = []
    FPR = []
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    lr_scores = lr.getScoresLogisticRegression(x, Dtest)
    scores = sr.calibrateScores(lr_scores, Ltest, lambd).flatten()
    sortedScores = np.sort(scores)
    
    for t in sortedScores:
        confusionMatrix = cf.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(scores, Ltest, t)
        FPRtemp, TPRtemp = cf.computeFPRTPR(prior, 1, 1, confusionMatrix)
        TPR.append(TPRtemp)
        FPR.append(FPRtemp)
    
    return (TPR, FPR)

def computeGMM(D, L, Dtest, Ltest, lambd, prior = 0.5):
    components = 4 # 16 components 
    
    print("GMM PCA m = 7 with", 2**components, "components")
        
    TPR = []
    FPR = []
    GMM0, GMM1 = GMM.trainGaussianClassifier(D, L, components)
    
    GMM_scores = GMM.getScoresGaussianClassifier(Dtest, GMM0, GMM1)
    scores = sr.calibrateScores(GMM_scores, Ltest, lambd, prior).flatten()
    sortedScores = np.sort(scores)
    
    for t in sortedScores:
        confusionMatrix = cf.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(scores, Ltest, t)
        FPRtemp, TPRtemp = cf.computeFPRTPR(prior, 1, 1, confusionMatrix)
        TPR.append(TPRtemp)
        FPR.append(FPRtemp)
        
    return (TPR, FPR)

def computeROC(D, L, Dtest, Ltest):
    
    lambd = 1e-4
    prior = 0.5
    

    PCA7 = dr.PCA(D, L, 7)
    PCA7_test = dr.PCA(Dtest, Ltest, 7)
    
    TPR_tiedcov, FPR_tiedcov = computeTiedCov(PCA7, L, PCA7_test, Ltest, lambd, prior)
    TPR_lr, FPR_lr = computeLR(PCA7, L, PCA7_test, Ltest, lambd, prior)
    TPR_gmm, FPR_gmm = computeGMM(D, L, Dtest, Ltest, lambd, prior)
    
    
    plot.plotROC(TPR_tiedcov, FPR_tiedcov, TPR_lr, FPR_lr, TPR_gmm, FPR_gmm, "./ROC/rocplot.png")
    
    return
