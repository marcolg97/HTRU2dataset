# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""

import numpy as np
import utils
import gaussianClassifier as gc
import scipy.special as scs
import dimensionality_reduction as dr
import folds
import cost_functions as cf
import plot

def trainGaussianClassifier(D, L, components):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    # GMM(1.0, mean, covariance) = GMM(w, mu, sigma)
    GMM0_start = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D0).reshape((D0.shape[0], D0.shape[0]))))]
    GMM1_start = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D1).reshape((D1.shape[0], D1.shape[0]))))]
    
    GMM0 = LBGalgorithm (GMM0_start, D0, components, mode = "full")
    GMM1 = LBGalgorithm (GMM1_start, D1, components, mode = "full")
    
    return GMM0, GMM1

def getScoresGaussianClassifier(X, GMM0, GMM1):
    LS0 = computeLogLikelihood(X, GMM0)
    LS1 = computeLogLikelihood(X, GMM1)
        
    llr = LS1-LS0
    return llr

def trainNaiveBayes(D, L, components):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    # GMM(1.0, mean, covariance) = GMM(w, mu, sigma)
    GMM0_start = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D0)*np.eye( D0.shape[0]).reshape((D0.shape[0]),D0.shape[0])))]
    GMM1_start = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainEigenvaluesCovariance(np.cov(D1)*np.eye( D1.shape[0]).reshape((D1.shape[0]),D1.shape[0])))]
    
    GMM0 = LBGalgorithm (GMM0_start, D0, components, mode = "diag")
    GMM1 = LBGalgorithm (GMM1_start, D1, components, mode = "diag")
    
    return GMM0, GMM1

def getScoresNaiveBayes(X, GMM0, GMM1):
    LS0 = computeLogLikelihood(X, GMM0)
    LS1 = computeLogLikelihood(X, GMM1)
        
    llr = LS1-LS0
    return llr

def trainTiedCov(D, L, components):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    sigma0 =  np.cov(D0).reshape((D0.shape[0], D0.shape[0]))
    sigma1 =  np.cov(D1).reshape((D1.shape[0], D1.shape[0]))
    
    sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*sigma0+D[:, L == 1].shape[1]*sigma1)
    # GMM(1.0, mean, covariance) = GMM(w, mu, sigma)
    GMM0_start = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainEigenvaluesCovariance(sigma))]
    GMM1_start = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainEigenvaluesCovariance(sigma))]
    
    GMM0 = LBGalgorithm (GMM0_start, D0, components, mode = "tied")
    GMM1 = LBGalgorithm (GMM1_start, D1, components, mode = "tied")
    
    return GMM0, GMM1

def getScoresTiedCov(X, GMM0, GMM1):
    LS0 = computeLogLikelihood(X, GMM0)
    LS1 = computeLogLikelihood(X, GMM1)
        
    llr = LS1-LS0
    return llr

def findGMMComponents(D, L, maxComp=7):
    '''
    Plot graphs for GMM full covariance, diagonal covariance and tied full covariance 
    using grid search in order to configure optimal parameters for the number of components

    Parameters
    ----------
    D : dataset
    L : label of the dataset
    maxComp : int, number of components to try  
        DESCRIPTION. The default is 7.

    Returns
    -------
    None.

    '''
    
    PCA7 = dr.PCA(D, L, 7)
    PCA6 = dr.PCA(D, L, 6)
    
       
    print("\n\n----------------Full-Cov-----------------------------------\n\n")
    
    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    print("NO PCA - SINGLE FOLD \n")
    minDCF_noPCA_singlefold = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresGaussianClassifier(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_noPCA_singlefold.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for noPCA-singlefold-fullcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_noPCA_singlefold, 
                 "GMM components", "./GMM/noPCA-singlefold-fullcov.png", base = 2)
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    print("NO PCA - K FOLD \n")
    minDCF_noPCA_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_noPCA_kfold.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for noPCA-kfold-fullcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_noPCA_kfold, 
                 "GMM components", "./GMM/noPCA-kfold-fullcov.png", base = 2)
    
    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    print("PCA m = 7 - SINGLE FOLD \n")
    minDCF_PCA7_singlefold = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresGaussianClassifier(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_PCA7_singlefold.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA7-singlefold-fullcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA7_singlefold, 
                 "GMM components", "./GMM/PCA7-singlefold-fullcov.png", base = 2)
    
    
    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    print("PCA (m=7) - K FOLD \n")
    minDCF_PCA7_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_PCA7_kfold.append(cost)
            print("component:", 2**(component), "cost:", cost)
                  
    print("\n\nPlot done for PCA7-kfold-fullcov.png \n\n")  
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA7_kfold, 
                 "GMM components", "./GMM/PCA7-kfold-fullcov.png", base = 2)
    
    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    print("PCA m = 6 - SINGLE FOLD \n")
    minDCF_PCA6_singlefold = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresGaussianClassifier(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_PCA6_singlefold.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA6-singlefold-fullcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA6_singlefold, 
                 "GMM components", "./GMM/PCA6-singlefold-fullcov.png", base = 2)
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    print("PCA (m=6) - K FOLD \n")
    minDCF_PCA6_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_PCA6_kfold.append(cost)
            print("component:", 2**(component), "cost:", cost)
                  
    print("\n\nPlot done for PCA6-kfold-fullcov.png \n\n")  
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA6_kfold, 
                 "GMM components", "./GMM/PCA6-kfold-fullcov.png", base = 2)
    
    
    print("\n\n----------------Diag-Cov-----------------------------------\n\n")
    
    
    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    print("NO PCA - SINGLE FOLD \n")
    minDCF_noPCA_singlefold_diag = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainNaiveBayes(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresNaiveBayes(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_noPCA_singlefold_diag.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for noPCA-singlefold-diagcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_noPCA_singlefold_diag, 
                 "GMM components", "./GMM/noPCA-singlefold-diagcov.png", base = 2)
    
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    print("no PCA - K FOLD \n")
    minDCF_noPCA_kfold_diag = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for component in range(maxComp):
            
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_noPCA_kfold_diag.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for noPCA-kfold-diagcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_noPCA_kfold_diag, "GMM components", 
                 "./GMM/noPCA-kfold-diagcov.png", base = 2)
    
    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    print("PCA m = 7 - SINGLE FOLD \n")
    minDCF_PCA7_singlefold_diag = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainNaiveBayes(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresNaiveBayes(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_PCA7_singlefold_diag.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA7-singlefold-diagcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA7_singlefold_diag, 
                 "GMM components", "./GMM/PCA7-singlefold-diagcov.png", base = 2)
          
    
    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    print("PCA m = 7 - K FOLD \n")
    minDCF_PCA7_kfold_diag = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for component in range(maxComp):
           
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_PCA7_kfold_diag.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA7-kfold-diagcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA7_kfold_diag, 
                 "GMM components", "./GMM/PCA7-kfold-diagcov.png", base = 2)
    
    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    print("PCA m = 6 - SINGLE FOLD \n")
    minDCF_PCA6_singlefold_diag = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainNaiveBayes(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresNaiveBayes(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_PCA6_singlefold_diag.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA6-singlefold-diagcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA6_singlefold_diag, 
                 "GMM components", "./GMM/PCA6-singlefold-diagcov.png", base = 2)
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    print("PCA (m=6) - K FOLD \n")
    minDCF_PCA6_kfold_diag = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_PCA6_kfold_diag.append(cost)
            print("component:", 2**(component), "cost:", cost)
                  
    print("\n\nPlot done for PCA6-kfold-diagcov.png \n\n")  
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA6_kfold_diag, 
                 "GMM components", "./GMM/PCA6-kfold-diagcov.png", base = 2)
    
    
    print("\n\n----------------Tied-Cov-----------------------------------\n\n")
    
    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    print("NO PCA - SINGLE FOLD \n")
    minDCF_noPCA_singlefold_tied = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainTiedCov(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresTiedCov(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_noPCA_singlefold_tied.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for noPCA-singlefold-tiedcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_noPCA_singlefold_tied, 
                 "GMM components", "./GMM/noPCA-singlefold-tiedcov.png", base = 2)
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    print("NO PCA - K FOLD \n")
    minDCF_noPCA_kfold_tied = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresTiedCov(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_noPCA_kfold_tied.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for noPCA-kfold-tiedcov.png \n\n")             
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_noPCA_kfold_tied, "GMM components", 
                 "./GMM/noPCA-kfold-tiedcov.png", base = 2)
    
    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    print("PCA m = 7 - SINGLE FOLD \n")
    minDCF_PCA7_singlefold_tied = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainTiedCov(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresTiedCov(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_PCA7_singlefold_tied.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA7-singlefold-tiedcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA7_singlefold_tied, 
                 "GMM components", "./GMM/PCA7-singlefold-tiedcov.png", base = 2)
              
    
    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    print("PCA m=7 - K FOLD \n")
    minDCF_PCA7_kfold_tied = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresTiedCov(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_PCA7_kfold_tied.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA7-kfold-tiedcov.png \n\n")  
    plot.plotDCF([2**(component+1) for component in range(maxComp)], minDCF_PCA7_kfold_tied, "GMM components", 
                 "./GMM/PCA7-kfold-tiedcov.png", base = 2)
    
    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    print("PCA m = 6 - SINGLE FOLD \n")
    minDCF_PCA6_singlefold_tied = []
     
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            GMM0, GMM1 = trainTiedCov(DTR, LTR, component)
            cost = cf.minimum_detection_costs(getScoresTiedCov(DEV, GMM0, GMM1) , LEV, model[0], model[1], model[2])
    
            minDCF_PCA6_singlefold_tied.append(cost)
            print("component:", 2**(component), "cost:", cost)
    
    print("\n\nPlot done for PCA6-singlefold-tiedcov.png \n\n")        
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA6_singlefold_tied, 
                 "GMM components", "./GMM/PCA6-singlefold-tiedcov.png", base = 2)
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    print("PCA (m=6) - K FOLD \n")
    minDCF_PCA6_kfold_tied = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    
    for model in utils.models:
        print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for component in range(maxComp):
            
            scores = []
            for singleKFold in allKFolds:
                GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], component)
                scores.append(getScoresTiedCov(singleKFold[2], GMM0, GMM1))
            
            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            
            minDCF_PCA6_kfold_tied.append(cost)
            print("component:", 2**(component), "cost:", cost)
                  
    print("\n\nPlot done for PCA6-kfold-tiedcov.png \n\n")  
    plot.plotDCF([2**(component) for component in range(maxComp)], minDCF_PCA6_kfold_tied, 
                 "GMM components", "./GMM/PCA6-kfold-tiedcov.png", base = 2)
    
    
    print("\n\nFINISH PLOTS FOR GMM")
    
    return

def computeGMM(D, L, components, mode = "full"):
    '''
    

    Parameters
    ----------
    D : dataset
    L : label of the dataset
    components : int, number of optimal components found before
    mode : define it to print value for full-cov, diag-cov or tied-full-cov
        DESCRIPTION. The default is "full".

    Returns
    -------
    None.

    '''

    PCA7 = dr.PCA(D, L, 7)
    PCA6 = dr.PCA(D, L, 6)
    
    allKFoldsnoPCA, evaluationLabelsnoPCA = folds.Kfold_without_train(D, L)
    allKFoldsPCA7, evaluationLabelsPCA7 = folds.Kfold_without_train(PCA7, L)
    allKFoldsPCA6, evaluationLabelsPCA6 = folds.Kfold_without_train(PCA6, L)
    
    if(mode == "full"):
        print("\nGMM FULL-COV \n")
    elif(mode == "diag"):
        print("\nGMM DIAG-COV \n")
    elif(mode == "tied"):
        print("\nGMM TIED FULL-COV \n")

    
    # ------------------- NO PCA ---------------

    print("\nNO PCA - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models:
        if(mode == "full"):
            GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, components)        
            score = getScoresGaussianClassifier(DEV, GMM0, GMM1)
        if(mode == "diag"):
            GMM0, GMM1 = trainNaiveBayes(DTR, LTR, components)
            score = getScoresNaiveBayes(DEV, GMM0, GMM1)
        if(mode == "tied"):
            GMM0, GMM1 = trainTiedCov(DTR, LTR, components)
            score = getScoresTiedCov(DEV, GMM0, GMM1)
        
        minDCF = cf.minimum_detection_costs(score, LEV, model[0], model[1], model[2])
        print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF)
    
    
    print("\nNO PCA - K FOLD \n")
    for model in utils.models:
        scores = []
        for singleKFold in allKFoldsnoPCA:
            if(mode == "full"):
                GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
                scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
            if(mode == "diag"):
                GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
            if(mode == "tied"):
                GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresTiedCov(singleKFold[2], GMM0, GMM1))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsnoPCA, model[0], model[1], model[2])
        print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 7 ---------------
    
    print("\nPCA m=7 - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models:
        if(mode == "full"):
            GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, components)        
            score = getScoresGaussianClassifier(DEV, GMM0, GMM1)
        if(mode == "diag"):
            GMM0, GMM1 = trainNaiveBayes(DTR, LTR, components)
            score = getScoresNaiveBayes(DEV, GMM0, GMM1)
        if(mode == "tied"):
            GMM0, GMM1 = trainTiedCov(DTR, LTR, components)
            score = getScoresTiedCov(DEV, GMM0, GMM1)
        
        minDCF = cf.minimum_detection_costs(score, LEV, model[0], model[1], model[2])
        print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF)
    
    
    print("\nPCA m=7 - K FOLD \n")
    for model in utils.models:
        scores = []
        for singleKFold in allKFoldsPCA7:
            if(mode == "full"):
                GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
                scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
            if(mode == "diag"):
                GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
            if(mode == "tied"):
                GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresTiedCov(singleKFold[2], GMM0, GMM1))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA7, model[0], model[1], model[2])
        print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF)
     
    # -------------- PCA m = 6 ---------------
    
    print("\nPCA m=6 - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models:
        if(mode == "full"):
            GMM0, GMM1 = trainGaussianClassifier(DTR, LTR, components)        
            score = getScoresGaussianClassifier(DEV, GMM0, GMM1)
        if(mode == "diag"):
            GMM0, GMM1 = trainNaiveBayes(DTR, LTR, components)
            score = getScoresNaiveBayes(DEV, GMM0, GMM1)
        if(mode == "tied"):
            GMM0, GMM1 = trainTiedCov(DTR, LTR, components)
            score = getScoresTiedCov(DEV, GMM0, GMM1)
        
        minDCF = cf.minimum_detection_costs(score, LEV, model[0], model[1], model[2])
        print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF)
    
    
    print("\nPCA m=6 - K FOLD \n")
    for model in utils.models:
        scores = []
        for singleKFold in allKFoldsPCA6:
            if(mode == "full"):
                GMM0, GMM1 = trainGaussianClassifier(singleKFold[1], singleKFold[0], components)        
                scores.append(getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
            if(mode == "diag"):
                GMM0, GMM1 = trainNaiveBayes(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
            if(mode == "tied"):
                GMM0, GMM1 = trainTiedCov(singleKFold[1], singleKFold[0], components)
                scores.append(getScoresTiedCov(singleKFold[2], GMM0, GMM1))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA6, model[0], model[1], model[2])
        print("components = ", 2**components, "application with prior:", model[0], "minDCF = ", minDCF)
            
    return

def constrainEigenvaluesCovariance(sigma, psi = 0.01):

    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, utils.mcol(s)*U.T)
    return sigma

def LBGalgorithm(GMM, X, iterations, mode = "full"):
    # estimate parameters for the initial GMM(1.0, mu, sigma)
    GMM = EMalgorithm(X, GMM, mode = mode)
    for i in range(iterations):
        # estimate new parameters after the split
        GMM = split(GMM)
        GMM = EMalgorithm(X, GMM, mode = mode)
    return GMM

def split(GMM, alpha = 0.1):
    splittedGMM = []
    # we split in 2 parts each component of the GMM
    for i in range(len(GMM)):  
        weight = GMM[i][0]
        mean = GMM[i][1]
        sigma = GMM[i][2]
        # find the leading eigenvector of sigma
        U, s, Vh = np.linalg.svd(sigma)
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((weight/2, mean + d, sigma))
        splittedGMM.append((weight/2, mean - d, sigma))
    return splittedGMM

def EMalgorithm(X, gmm, delta = 10**(-6), mode = "full"):
    flag = True
    while(flag):
        # Compute log marginal density with initial parameters
        
        #S = logpdf_GMM(X, gmm)
        
        S = joint_log_density_GMM(logpdf_GMM(X, gmm), gmm)
        
        #logmarg = marginalDensityGMM(S, gmm)
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm))
        
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = log_likelihood_GMM(logmarg, X)
        
        # ------ E-step ----------
        # Compute the posterior probability for each component of the GMM for each sample
        posteriorProbability = np.exp(S - logmarg.reshape(1, logmarg.size))
        
        # ------ M-step ----------
        (w, mu, cov) = Mstep(X, S, posteriorProbability, mode)
        
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
       
        # Compute the new log densities and the new sub-class conditional densities
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )                                                                            
        loglikelihood2 = log_likelihood_GMM(logmarg, X)
        
        if (loglikelihood2 - loglikelihood1 < delta):
            flag = False
        if (loglikelihood2 - loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    return gmm

def Mstep(X, S, posterior, mode = "full"):
    Zg = np.sum(posterior, axis=1)  # 3
    
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        Sum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            Sum += posterior[g, i] * X[:, i]
        Fg[:, g] = Sum
    
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        Sum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            X_i = X[:, i].reshape((X.shape[0], 1))
            X_iT = X[:, i].reshape((1, X.shape[0]))
            Sum += posterior[g, i] * np.dot(X_i, X_iT)
        Sg[g] = Sum
    
    mu_new = Fg / Zg

    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu_new[:, g].reshape((X.shape[0], 1)),
                           mu_new[:, g].reshape((1, X.shape[0])))
    
    cov_new = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
   
    
    if(mode == "full"): 
        for g in range(S.shape[0]):        
            cov_new[g] = constrainEigenvaluesCovariance(cov_new[g])
    elif(mode == "diag"):
        for g in range(S.shape[0]):
            cov_new[g] = constrainEigenvaluesCovariance(cov_new[g] * np.eye(cov_new[g].shape[0]))
    elif(mode == "tied"):
        tsum = np.zeros((cov_new.shape[1], cov_new.shape[2]))
        for g in range(S.shape[0]):
            tsum += Zg[g]*cov_new[g]
        for g in range(S.shape[0]):
            cov_new[g] = constrainEigenvaluesCovariance(1/X.shape[1] * tsum)
            
    w_new = Zg/np.sum(Zg)
    
    return (w_new, mu_new, cov_new)


def logpdf_GMM(X, gmm):
    S = np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        # Compute log density
        S[i, :] = gc.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return S

def joint_log_density_GMM (S, gmm):
    
    for i in range(len(gmm)):
        # Add log of the prior of the corresponding component
        S[i, :] += np.log(gmm[i][0])
    return S

def marginal_density_GMM (S):
    return scs.logsumexp(S, axis = 0)


def log_likelihood_GMM(logmarg, X):
    return np.sum(logmarg)/X.shape[1]

def computeLogLikelihood(X, gmm):
    tempSum=np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        tempSum[i,:]= np.log(gmm[i][0]) + gc.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return scs.logsumexp(tempSum, axis=0)