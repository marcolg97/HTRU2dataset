# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""

import numpy as np
from itertools import repeat
import scipy.optimize as scopt

import dimensionality_reduction as dr
import utils
import folds
import cost_functions as cf
import plot

def trainLinearSVM(DTR, LTR, C = 1.0, K = 1.0, pi_T = None):
    
    if(pi_T == None):
        w = modifiedDualFormulation(DTR, LTR, C, K)
    else:
        w = modifiedDualFormulationBalanced(DTR, LTR, C, K, pi_T)
    return w

def getScoresLinearSVM(w, DEV, K = 1.0):
    DEV = np.vstack([DEV, np.zeros(DEV.shape[1]) + K])
    S = np.dot(w.T, DEV)
    return S

def trainPolynomialSVM(DTR, LTR, C = 1.0, K = 1.0, c = 0, d = 2):
    x = kernelPoly(DTR, LTR, K, C, d, c)
    return x

def getScoresPolynomialSVM(x, DTR, LTR, DEV, K = 1.0, c = 0, d = 2):
    S = np.sum(np.dot((x * LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DEV) + c)**d + K**2), axis=0)
    return S

def trainRBFKernel(DTR, LTR, gamma = 1.0, K = 1.0, C = 1.0):
    x = kernelRBF(DTR, LTR, K, C, gamma)
    return x

def getScoresRBFKernel(x, DTR, LTR, DEV, gamma = 1.0, K = 1.0):
    kernelFunction = np.zeros((DTR.shape[1], DEV.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DEV.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DEV[:, j], gamma, K**2)
    S=np.sum(np.dot((x*LTR).reshape(1, DTR.shape[1]), kernelFunction), axis=0)
    return S
    


def findUnbalancedC (D, L, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30)): 
    '''
    Plot graphs for SVM unbalanced using grid search in order to configure optimal parameters for k and C

    Parameters
    ----------
    D : dataset
    L : label of the dayaset
    rangeK : int, optional
        range of k values to try. The default is [1.0, 10.0].
    rangeC : int, optional
        range for C to try. The default is np.logspace(-5, -1, num=30).

    Returns
    -------
    None.

    '''
    
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
    
   
    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    print("NO PCA - SINGLE FOLD \n")
    minDCF_noPCA_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models: 
                
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            
            for C in rangeC:
                                
                w = trainLinearSVM(DTR, LTR, C, k)
    
                cost = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV, k), LEV, model[0], model[1], model[2])
                
                minDCF_noPCA_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
    
    print("\n\nPlot done for noPCA-singlefold.png \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_noPCA_singlefold, "C", "./SVM_unbalanced/noPCA-singlefold.png")
    print("\n\n")
    
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    print("NO PCA - K FOLD \n")
    minDCF_noPCA_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for model in utils.models:
        
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            for C in rangeC:
                scores = []
                
                for singleKFold in allKFolds:
                    w = trainLinearSVM(singleKFold[1], singleKFold[0], C, k)
                    scores.append(getScoresLinearSVM(w, singleKFold[2], k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_noPCA_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for noPCA-kfold.png \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_noPCA_kfold, "C", "./SVM_unbalanced/noPCA-kfold.png")
    print("\n\n")
    
    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    print("PCA m=7 - SINGLE FOLD \n")
    minDCF_PCA7_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models: 
                
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            
            for C in rangeC:
                                
                w = trainLinearSVM(DTR, LTR, C, k)
    
                cost = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV, k), LEV, model[0], model[1], model[2])
                
                minDCF_PCA7_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
    
    print("\n\nPlot done for PCA7-singlefold.png \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA7_singlefold, "C", "./SVM_unbalanced/PCA7-singlefold.png")
    print("\n\n")


    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    print("PCA m=7 - K FOLD \n")
    minDCF_PCA7_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    
    for model in utils.models:
        
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            for C in rangeC:
                scores = []
                
                for singleKFold in allKFolds:
                    w = trainLinearSVM(singleKFold[1], singleKFold[0], C, k)
                    scores.append(getScoresLinearSVM(w, singleKFold[2], k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA7_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA7-kfold.png \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA7_kfold, "C", "./SVM_unbalanced/PCA7-kfold.png")
    print("\n\n")

    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    print("PCA m=6 - SINGLE FOLD \n")
    minDCF_PCA6_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models: 
                
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            
            for C in rangeC:
                                
                w = trainLinearSVM(DTR, LTR, C, k)
    
                cost = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV, k), LEV, model[0], model[1], model[2])
                
                minDCF_PCA6_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
    
    print("\n\nPlot done for PCA6-singlefold.png \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA6_singlefold, "C", "./SVM_unbalanced/PCA6-singlefold.png")
    print("\n\n")
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    print("PCA m=6 - K FOLD \n")
    minDCF_PCA6_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    
    for model in utils.models:
        
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            for C in rangeC:
                scores = []
                
                for singleKFold in allKFolds:
                    w = trainLinearSVM(singleKFold[1], singleKFold[0], C, k)
                    scores.append(getScoresLinearSVM(w, singleKFold[2], k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA6_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA6-kfold.png \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA6_kfold, "C", "./SVM_unbalanced/PCA6-kfold.png")
    print("\n\n")
    
    print("\n\nFINISH PLOTS FOR SVM UNBALANCED")
    return

def findBalancedC (D, L, pi_T, rangeK=[1.0, 10.0], rangeC=np.logspace(-5, -1, num=30)):
    '''
    Plot graphs for SVM unbalanced using grid search in order to configure optimal parameters for k and C

    Parameters
    ----------
    D : dataset
    L : label of the dataset
    rangeK : int, optional
        range of k values to try. The default is [1.0, 10.0].
    rangeC : int, optional
        range for C to try. The default is np.logspace(-5, -1, num=30).

    Returns
    -------
    None.

    '''
    
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)

    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    print("NO PCA - SINGLE FOLD \n")
    minDCF_noPCA_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models: 
                
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            
            for C in rangeC:
                                
                w = trainLinearSVM(DTR, LTR, C, k, pi_T)
    
                cost = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV, k), LEV, model[0], model[1], model[2])
                
                minDCF_noPCA_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
    
    filename = "./SVM_balanced/noPCA-singlefold-" + str(pi_T) + ".png"
    print("\n\nPlot done for", filename, " \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_noPCA_singlefold, "C", filename)
    print("\n\n")
    
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    print("NO PCA - K FOLD \n")
    minDCF_noPCA_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for model in utils.models:
        
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            for C in rangeC:
                scores = []
                
                for singleKFold in allKFolds:
                    w = trainLinearSVM(singleKFold[1], singleKFold[0], C, k, pi_T)
                    scores.append(getScoresLinearSVM(w, singleKFold[2], k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_noPCA_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    filename = "./SVM_balanced/noPCA-kfold-" + str(pi_T) + ".png"
    print("\n\nPlot done for", filename, " \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_noPCA_singlefold, "C", filename)
    print("\n\n")
    
        
    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    print("PCA m=7 - SINGLE FOLD \n")
    minDCF_PCA7_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models: 
                
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            
            for C in rangeC:
                                
                w = trainLinearSVM(DTR, LTR, C, k, pi_T)
    
                cost = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV, k), LEV, model[0], model[1], model[2])
                
                minDCF_PCA7_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
    
    filename = "./SVM_balanced/PCA7-singlefold-" + str(pi_T) + ".png"
    print("\n\nPlot done for", filename, " \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA7_singlefold, "C", filename)
    print("\n\n")
    
    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    print("PCA m=7 - K FOLD \n")
    minDCF_PCA7_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    
    for model in utils.models:
        
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            for C in rangeC:
                scores = []
                
                for singleKFold in allKFolds:
                    w = trainLinearSVM(singleKFold[1], singleKFold[0], C, k, pi_T)
                    scores.append(getScoresLinearSVM(w, singleKFold[2], k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA7_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    filename = "./SVM_balanced/PCA7-kfold-" + str(pi_T) + ".png"            
    print("\n\nPlot done for", filename, " \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA7_kfold, "C", filename)
    print("\n\n")
               
    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    print("PCA m=6 - SINGLE FOLD \n")
    minDCF_PCA6_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models: 
                
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            
            for C in rangeC:
                                
                w = trainLinearSVM(DTR, LTR, C, k, pi_T)
    
                cost = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV, k), LEV, model[0], model[1], model[2])
                
                minDCF_PCA6_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
    
    filename = "./SVM_balanced/PCA6-singlefold-" + str(pi_T) + ".png"
    print("\n\nPlot done for", filename, " \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA6_singlefold, "C", filename)
    print("\n\n")
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    print("PCA m=6 - K FOLD \n")
    minDCF_PCA6_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    
    for model in utils.models:
        
        for k in rangeK:
            print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
            for C in rangeC:
                scores = []
                
                for singleKFold in allKFolds:
                    w = trainLinearSVM(singleKFold[1], singleKFold[0], C, k, pi_T)
                    scores.append(getScoresLinearSVM(w, singleKFold[2], k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA6_kfold.append(cost)
                print("C:", C, ", cost:", cost)
           
    filename = "./SVM_balanced/PCA6-kfold-" + str(pi_T) + ".png"
    print("\n\nPlot done for", filename, " \n\n")
    plot.plotDCF_SVM(rangeC, minDCF_PCA6_kfold, "C", filename)
    print("\n\n")
    
    print("\n\nFINISH PLOTS FOR SVM BALANCED")
    
    return

def computeLinearSVM(D, L, C, K=1.0, pi_T = None):
    '''
    Generate the result for the SVM table with balanced and unbalanced class and priors 0.5, 0.9 and 0.1

    Parameters
    ----------
    D : dataset
    L : label of the dataset
    C : value of C found before
    K : value of K found before
    pi_T : if None the linear SVM is unbalanced, otherwise it's balanced with the prior we pass

    Returns
    -------
    None.

    '''
    
    
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
    
    allKFoldsnoPCA, evaluationLabelsnoPCA = folds.Kfold_without_train(D, L)
    allKFoldsPCA7, evaluationLabelsPCA7 = folds.Kfold_without_train(PCA7, L)
    allKFoldsPCA6, evaluationLabelsPCA6 = folds.Kfold_without_train(PCA6, L)
        
    # ------------------- NO PCA ---------------
    print("NO PCA - SINGLE FOLD \n")
    
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        w = trainLinearSVM(DTR, LTR, C, K, pi_T)

        minDCF = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV) , LEV, model[0], model[1], model[2])
        
        print("C = ", C, "pi_T =", pi_T, "application with prior:", model[0], "minDCF = ", minDCF)
   

    print("NO PCA - K FOLD \n")
       
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsnoPCA:
            w = trainLinearSVM(singleKFold[1], singleKFold[0], C, K, pi_T)
            scores.append(getScoresLinearSVM(w, singleKFold[2]))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsnoPCA, model[0], model[1], model[2])
        
        print("C = ", C, "pi_T =", pi_T, "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 7 ---------------
    
    print("PCA m=7 - SINGLE FOLD \n")
    
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        w = trainLinearSVM(DTR, LTR, C, K, pi_T)

        minDCF = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV) , LEV, model[0], model[1], model[2])
        
        print("C = ", C, "pi_T =", pi_T, "application with prior:", model[0], "minDCF = ", minDCF)
        
    print("PCA m=7 - K FOLD \n")  
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsPCA7:
            w = trainLinearSVM(singleKFold[1], singleKFold[0], C, K, pi_T)
            scores.append(getScoresLinearSVM(w, singleKFold[2]))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA7, model[0], model[1], model[2])
        
        print("C = ", C, "pi_T =", pi_T, "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 6 ---------------
    print("PCA m=6 - SINGLE FOLD \n")  
    
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        w = trainLinearSVM(DTR, LTR, C, K, pi_T)

        minDCF = cf.minimum_detection_costs(getScoresLinearSVM(w, DEV) , LEV, model[0], model[1], model[2])
        
        print("C = ", C, "pi_T =", pi_T, "application with prior:", model[0], "minDCF = ", minDCF)
        
        
    print("PCA m=6 - K FOLD \n")  
    
    for model in utils.models:  
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsPCA6:
            w = trainLinearSVM(singleKFold[1], singleKFold[0], C, K, pi_T)
            scores.append(getScoresLinearSVM(w, singleKFold[2]))

        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA6, model[0], model[1], model[2])
        
        print("C = ", C, "pi_T =", pi_T, "application with prior:", model[0], "minDCF = ", minDCF)
            
    return

def findPolynomialC (D, L, rangeK=[0.0, 1.0], d=2, rangec=[0, 1, 15], rangeC=np.logspace(-5, -1, num=30)):
    '''
    Plot graphs for polynomial SVM using grid search in order to configure optimal parameters for c and C

    Parameters
    ----------
    D : dataset
    L : label
    rangeK : int, optional
         range of k values to try. The default is [0.0, 1.0].
    d : int, optional
        d. The default is 2.
    rangec : int, optional
        range for c values to try. The default is [0, 1, 15].
    rangeC : TYPE, optional
        range for C to try. The default is np.logspace(-5, -1, num=30).

    Returns
    -------
    None.

    '''
        
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
    
    model= utils.models[0] # Take only prior 0.5, cfp 1 and cfn 1
    
    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    print("NO PCA - SINGLE FOLD \n")
    minDCF_noPCA_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for c in rangec:
            print("c:", c)
            for C in rangeC:
                x = trainPolynomialSVM(DTR, LTR, C, k, c, d)
    
                cost = cf.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, k, c, d), LEV, model[0], model[1], model[2])
                
                minDCF_noPCA_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for noPCA-singlefold.png \n\n")            
    plot.plotFindC(rangeC, minDCF_noPCA_singlefold, "C", "./SVM_poly/noPCA-singlefold.png")
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    print("NO PCA - K FOLD \n")
    minDCF_noPCA_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for c in rangec:
            print("c:", c)
            for C in rangeC:
                scores = []
                for singleKFold in allKFolds:
                    x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, k, c, d)
    
                    scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], k, c, d))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_noPCA_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for noPCA-kfold.png \n\n")             
    plot.plotFindC(rangeC, minDCF_noPCA_kfold, "C", "./SVM_poly/noPCA-kfold.png")
    
    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    print("PCA m=7 - single FOLD \n")
    minDCF_PCA7_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for c in rangec:
            print("c:", c)
            for C in rangeC:
                x = trainPolynomialSVM(DTR, LTR, C, k, c, d)
    
                cost = cf.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, k, c, d), LEV, model[0], model[1], model[2])
                
                minDCF_PCA7_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA7-singlefold.png \n\n")              
    plot.plotFindC(rangeC, minDCF_PCA7_singlefold, "C", "./SVM_poly/PCA7-singlefold.png")
    
    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    print("PCA m=7 - K FOLD \n")
    minDCF_PCA7_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for c in rangec:
            print("c:", c)
            for C in rangeC:
                scores = []
                for singleKFold in allKFolds:
                    x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, k, c, d)
    
                    scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], k, c, d))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA7_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA7-kfold.png \n\n")              
    plot.plotFindC(rangeC, minDCF_PCA7_kfold, "C", "./SVM_poly/PCA7-kfold.png")
    
    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    print("PCA m=6 - single FOLD \n")
    minDCF_PCA6_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for c in rangec:
            print("c:", c)
            for C in rangeC:
                x = trainPolynomialSVM(DTR, LTR, C, k, c, d)
     
                cost = cf.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, k, c, d), LEV, model[0], model[1], model[2])
                
                minDCF_PCA6_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA6-singlefold.png \n\n")                
    plot.plotFindC(rangeC, minDCF_PCA6_singlefold, "C", "./SVM_poly/PCA6-singlefold.png")
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    print("PCA m=6 - k FOLD \n")
    minDCF_PCA6_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for c in rangec:
            print("c:", c)
            for C in rangeC:
                scores = []
                for singleKFold in allKFolds:
                    x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, k, c, d)
    
                    scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], k, c, d))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA6_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA6-kfold.png \n\n")               
    plot.plotFindC(rangeC, minDCF_PCA6_kfold, "C", "./SVM_poly/PCA6-kfold.png")
    
    print("\n\nFINISH PLOTS FOR SVM POLYNOMIAL")
    
    return

def computePolynomialSVM(D, L, C, c, K = 1.0, d = 2):
    '''
    Generate the result for the polynomial SVM table with priors 0.5, 0.9 and 0.1

    Parameters
    ----------
    D : dataset
    L : label of the dataset
    C : value of C we found before
    c : value of c we found before
    K : value of K we found before
    d : value of d we decide before

    Returns
    -------
    None.

    '''
    
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
    
    allKFoldsnoPCA, evaluationLabelsnoPCA = folds.Kfold_without_train(D, L)
    allKFoldsPCA7, evaluationLabelsPCA7 = folds.Kfold_without_train(PCA7, L)
    allKFoldsPCA6, evaluationLabelsPCA6 = folds.Kfold_without_train(PCA6, L)
    
    
    # ------------------- NO PCA ---------------

    print("NO PCA - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        x = trainPolynomialSVM(DTR, LTR, C, K, c, d)
        minDCF = cf.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, K, c, d), LEV, model[0], model[1], model[2])
        print("C = ", C, "c =", c, "application with prior:", model[0], "minDCF = ", minDCF)
    
    
    print("NO PCA - K FOLD \n")
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsnoPCA:
            
            x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, K, c, d)   
            scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], K, c, d))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsnoPCA, model[0], model[1], model[2])
        
        print("C = ", C, "c =", c, "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 7 ---------------
    
    print("PCA m=7 - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        x = trainPolynomialSVM(DTR, LTR, C, K, c, d)
        
        minDCF = cf.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, K, c, d), LEV, model[0], model[1], model[2])
        print("C = ", C, "c =", c, "application with prior:", model[0], "minDCF = ", minDCF)
    
    
    print("PCA m=7 - K FOLD \n")
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsPCA7:
           x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, K, c, d)
           scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], K, c, d))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA7, model[0], model[1], model[2])
        
        print("C = ", C, "c =", c, "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 6 ---------------
    
    print("PCA m=6 - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        x = trainPolynomialSVM(DTR, LTR, C, K, c, d)
        
        minDCF = cf.minimum_detection_costs(getScoresPolynomialSVM(x, DTR, LTR, DEV, K, c, d), LEV, model[0], model[1], model[2])
        print("C = ", C, "c =", c, "application with prior:", model[0], "minDCF = ", minDCF)
        
    
    print("PCA m=6 - K FOLD \n")
    for model in utils.models:  
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsPCA6:
            x = trainPolynomialSVM(singleKFold[1], singleKFold[0], C, K, c, d)
            scores.append(getScoresPolynomialSVM(x, singleKFold[1], singleKFold[0], singleKFold[2], K, c, d))

        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA6, model[0], model[1], model[2])
        
        print("C = ", C, "c =", c, "application with prior:", model[0], "minDCF = ", minDCF)
            
    return


def findRBFKernelC (D, L, rangeK = [0.0, 1.0], rangeC = np.logspace(-3, 3, num=30), rangeGamma = [10**(-4),10**(-3)]):
    '''
    Plot graphs for kernel RBF SVM using grid search in order to configure optimal parameters for gamma and C

    Parameters
    ----------
    D : dataset
    L : label of the dataset
    rangeK : int, optional
        range of k values to try. The default is [0.0, 1.0].
    rangeC : int, optional
        range for C to try. The default is np.logspace(-3, 3, num=30).
    rangeGamma : int, optional
        range for gamma to try. The default is [10**(-4),10**(-3)].

    Returns
    -------
    None.

    '''
    
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
        
    model = utils.models[0] # Take only prior 0.5, cfp 1 and cfn 1
    
    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    
    print("NO PCA - SINGLE FOLD \n")
    minDCF_noPCA_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for gamma in rangeGamma:
            print("gamma:", gamma)
            for C in rangeC:
                x = trainRBFKernel(DTR, LTR, gamma, k, C)
                cost = cf.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, k), LEV, model[0], model[1], model[2])
                
                minDCF_noPCA_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for noPCA-singlefold.png \n\n")    
    plot.plotDCF_RBF(rangeC, minDCF_noPCA_singlefold, "C", "./SVM_RBF/noPCA-singlefold.png")
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    print("NO PCA - K FOLD \n")
    minDCF_noPCA_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for gamma in rangeGamma:
            print("gamma:", gamma)
            for C in rangeC:
                scores = []
                for singleKFold in allKFolds:
                    x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, k, C)
    
                    scores.append(getScoresRBFKernel(x, singleKFold[1], singleKFold[0], singleKFold[2], gamma, k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                
                minDCF_noPCA_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for noPCA-kfold.png \n\n")    
    plot.plotDCF_RBF(rangeC, minDCF_noPCA_kfold, "C", "./SVM_RBF/noPCA-kfold.png")
    

    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    print("PCA m=7 - SINGLE FOLD \n")
    minDCF_PCA7_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for gamma in rangeGamma:
            print("gamma:", gamma)
            for C in rangeC:
                x = trainRBFKernel(DTR, LTR, gamma, k, C)
                cost = cf.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, k), LEV, model[0], model[1], model[2])
                
                minDCF_PCA7_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA7-singlefold.png \n\n")   
    plot.plotDCF_RBF(rangeC, minDCF_PCA7_singlefold, "C", "./SVM_RBF/PCA7-singlefold.png")
    
    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    print("PCA m=7 - K FOLD \n")
    minDCF_PCA7_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for gamma in rangeGamma:
            print("gamma:", gamma)
            for C in rangeC:
                scores = []
                for singleKFold in allKFolds:
                    x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, k, C)
    
                    scores.append(getScoresRBFKernel(x, singleKFold[1], singleKFold[0], singleKFold[2], gamma, k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA7_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA7-kfold.png \n\n")  
    plot.plotDCF_RBF(rangeC, minDCF_PCA7_kfold, "C", "./SVM_RBF/PCA7-kfold.png")
    
    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    print("PCA m=6 - SINGLE FOLD \n")
    minDCF_PCA6_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for gamma in rangeGamma:
            print("gamma:", gamma)
            for C in rangeC:
                x = trainRBFKernel(DTR, LTR, gamma, k, C)
                cost = cf.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, k), LEV, model[0], model[1], model[2])
                
                minDCF_PCA6_singlefold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA6-singlefold.png \n\n")  
    plot.plotDCF_RBF(rangeC, minDCF_PCA6_singlefold, "C", "./SVM_RBF/PCA6-singlefold.png")
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    print("PCA m=6 - K FOLD \n")
    minDCF_PCA6_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    
    for k in rangeK:
        print("\nStarting draw with k", k, ", prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for gamma in rangeGamma:
            print("gamma:", gamma)
            for C in rangeC:
                scores = []
                for singleKFold in allKFolds:
                    x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, k, C)
    
                    scores.append(getScoresRBFKernel(x, singleKFold[1], singleKFold[0], singleKFold[2], gamma, k))
    
                scores=np.hstack(scores)
                cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
                minDCF_PCA6_kfold.append(cost)
                print("C:", C, ", cost:", cost)
                
    print("\n\nPlot done for PCA6-kfold.png \n\n")  
    plot.plotDCF_RBF(rangeC, minDCF_PCA6_kfold, "C", "./SVM_RBF/PCA6-kfold.png")
    
    return

def computeRBFKernel(D, L, C, gamma, K = 1.0):
    '''
    Generate the result for the kernel RBF SVM table with priors 0.5, 0.9 and 0.1

    Parameters
    ----------
    D : dataset
    L : label of the dataset
    C : value of C we found before
    gamma : value of gamma we found before
    K : value of K we found before

    Returns
    -------
    None.

    '''

    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
    
    allKFoldsnoPCA, evaluationLabelsnoPCA = folds.Kfold_without_train(D, L)
    allKFoldsPCA7, evaluationLabelsPCA7 = folds.Kfold_without_train(PCA7, L)
    allKFoldsPCA6, evaluationLabelsPCA6 = folds.Kfold_without_train(PCA6, L)
       
    # ------------------- NO PCA ---------------
    print("NO PCA - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        x = trainRBFKernel(DTR, LTR, gamma, K, C)
        minDCF = cf.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, K) , LEV, model[0], model[1], model[2])
        print("C = ", C, "gamma =", gamma, "application with prior:", model[0], "minDCF = ", minDCF)


    print("NO PCA - K FOLD \n")
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        scores = []
        
        for singleKFold in allKFoldsnoPCA:
            x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, K, C)
            scores.append(getScoresRBFKernel(x,  singleKFold[1], singleKFold[0], singleKFold[2], gamma, K))
          
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsnoPCA, model[0], model[1], model[2])
        
        print("C = ", C, "gamma =", gamma, "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 7 ---------------
    print("PCA m=7 - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        x = trainRBFKernel(DTR, LTR, gamma, K, C)
        minDCF = cf.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, K) , LEV, model[0], model[1], model[2])
        print("C = ", C, "gamma =", gamma, "application with prior:", model[0], "minDCF = ", minDCF)
    
    
    print("PCA m=7 - K FOLD \n")
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsPCA7:
           x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, K, C)
           scores.append(getScoresRBFKernel(x,  singleKFold[1], singleKFold[0], singleKFold[2], gamma, K))
    
        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA7, model[0], model[1], model[2])
        
        print("C = ", C, "gamma =", gamma, "application with prior:", model[0], "minDCF = ", minDCF)
  
    # -------------- PCA m = 6 ---------------
    
    print("PCA m=6 - SINGLE FOLD \n")
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    
    for model in utils.models:
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        x = trainRBFKernel(DTR, LTR, gamma, K, C)
        minDCF = cf.minimum_detection_costs(getScoresRBFKernel(x, DTR, LTR, DEV, gamma, K) , LEV, model[0], model[1], model[2])
        print("C = ", C, "gamma =", gamma, "application with prior:", model[0], "minDCF = ", minDCF)
    
    
    print("PCA m=6 - K FOLD \n")
    for model in utils.models:  
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        scores = []
        for singleKFold in allKFoldsPCA6:
            x = trainRBFKernel(singleKFold[1], singleKFold[0], gamma, K, C)
            scores.append(getScoresRBFKernel(x,  singleKFold[1], singleKFold[0], singleKFold[2], gamma, K))

        scores=np.hstack(scores)
        minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA6, model[0], model[1], model[2])
        
        print("C = ", C, "gamma =", gamma, "application with prior:", model[0], "minDCF = ", minDCF)

    return

def modifiedDualFormulation(DTR, LTR, C, K):
    # Compute the D matrix for the extended training set with K=1
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])
    
    # Compute the H matrix exploiting broadcasting
    Gij = np.dot(D.T, D)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    # We use three different values for hyperparameter C
    
    w = primalLossDualLossDualityGapErrorRate(DTR, C, Hij, LTR, D, K)

    return w

def modifiedDualFormulationBalanced(DTR, LTR, C, K, piT):
    # Compute the D matrix for the extended training set
   
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])

    # Compute the H matrix 
    Gij = np.dot(D.T, D)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    
    
    C1 = C*piT/(DTR[:,LTR == 1].shape[1]/DTR.shape[1])
    C0 = C*(1-piT)/(DTR[:,LTR == 0].shape[1]/DTR.shape[1])
    
    boxConstraint = []
    for i in range(DTR.shape[1]):
        if LTR[i]== 1:
            boxConstraint.append ((0,C1))
        elif LTR[i]== 0:
            boxConstraint.append ((0,C0))
    
    (x, f, d) = scopt.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM, np.zeros(DTR.shape[1]), args=(Hij,), bounds=boxConstraint, factr=1.0)
    return np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)

def primalLossDualLossDualityGapErrorRate(DTR, C, Hij, LTR, D, K):
    #[ (0, C), (0, C), ..., (0, C)]
    boxConstraint = list(repeat((0, C), DTR.shape[1]))
    
    (x, f, d) = scopt.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM, np.zeros(DTR.shape[1]), args=(Hij,), bounds=boxConstraint, factr=1.0)
    
    # Now we can recover the primal solution
    alfa = x
    # All xi are inside D
    w = np.sum((alfa*LTR).reshape(1, DTR.shape[1])*D, axis=1)

    return w

def dualObjectiveOfModifiedSVM(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2)*np.dot(np.dot(alpha.T, H), alpha)-np.dot(alpha.T, np.ones(H.shape[1])), grad)

def kernelPoly(DTR, LTR, K, C, d, c):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = (np.dot(DTR.T, DTR)+c)**d+ K**2
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    x = dualLossErrorRatePoly(DTR, C, Hij, LTR, K, d, c)
    return x

def dualLossErrorRatePoly(DTR, C, Hij, LTR, K, d, c):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scopt.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return x

def kernelRBF(DTR, LTR, K, C, gamma):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTR[:, j], gamma, K)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    x = dualLossErrorRateRBF(DTR, C, Hij, LTR, K, gamma)
    return x

def RBF(x1, x2, gamma, K):
    return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))+K**2

def dualLossErrorRateRBF(DTR, C, Hij, LTR, K, gamma):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scopt.fmin_l_bfgs_b(dualObjectiveOfModifiedSVM,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return x