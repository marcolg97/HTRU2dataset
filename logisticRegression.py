# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""

import numpy as np
import scipy.optimize as scopt
import folds
import cost_functions as cf
import dimensionality_reduction as dr
import utils
import plot

def trainLogisticRegression(DTR, LTR, l, prior):
    (x, f, d) = scopt.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), args=(DTR, LTR, l, prior), approx_grad=True)
    return x,f,d

def logreg_obj(v, DTR, LTR, l, prior):
    w, b = v[0:-1], v[-1]
    j = J_balanced(w, b, DTR, LTR, l, prior)
    return j

def J_balanced(w, b, DTR, LTR, l, prior):
    normTerm = l/2*(np.linalg.norm(w)**2)
    sumTermTrueClass = 0
    sumTermFalseClass = 0
    for i in range(DTR.shape[1]):
        if LTR[i]==1:
            # if c_i = 1 -> z_i = 1
            sumTermTrueClass += np.log1p(np.exp(-np.dot(w.T, DTR[:, i])-b)) 
        else:
            # if c_i = 0 -> z_i = -1
            sumTermFalseClass += np.log1p(np.exp(np.dot(w.T, DTR[:, i])+b)) 
    j = normTerm + (prior/DTR[:, LTR==1].shape[1])*sumTermTrueClass + ((1-prior)/DTR[:, LTR==0].shape[1])*sumTermFalseClass
    return j

def getScoresLogisticRegression(x, DEV):
    S = np.dot(x[0:-1], DEV) + x[-1]
    return S
    

def findBestLambda(D,L):
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
    lambdas = np.logspace(-5, 5, num=50)
     
    
    # ----------------------------- NO PCA - SINGLE FOLD -----------------------------
    minDCF_noPCA_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            x,f,d = trainLogisticRegression(DTR, LTR, l, 0.5)
            cost = cf.minimum_detection_costs(getScoresLogisticRegression(x, DEV), LEV, model[0], model[1], model[2])
            minDCF_noPCA_singlefold.append(cost)
            print("Lambda:", l, ", cost:", cost)
    plot.plotDCF(lambdas, minDCF_noPCA_singlefold, "λ", "./logisticRegression/noPCA-singlefold.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    
    # ----------------------------- NO PCA - K FOLD -----------------------------
    minDCF_noPCA_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            scores = []
            for singleKFold in allKFolds:
                x, f, d = trainLogisticRegression(singleKFold[1], singleKFold[0], l, 0.5)
                scores.append(getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_noPCA_kfold.append(cost)
            print("Lambda:", l, ", cost:", cost)
    plot.plotDCF(lambdas, minDCF_noPCA_kfold, "λ", "./logisticRegression/noPCA-kfold.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    
    # ----------------------------- PCA m = 7 - SINGLE FOLD -----------------------------
    minDCF_PCA7_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA7, L)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            x,f,d = trainLogisticRegression(DTR, LTR, l, 0.5)
            cost = cf.minimum_detection_costs(getScoresLogisticRegression(x, DEV), LEV, model[0], model[1], model[2])
            minDCF_PCA7_singlefold.append(cost)
            print("Lambda:", l, ", cost:", cost)
    plot.plotDCF(lambdas, minDCF_PCA7_singlefold, "λ", "./logisticRegression/PCA7-singlefold.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    
    # ----------------------------- PCA m = 7 - K FOLD -----------------------------
    minDCF_PCA7_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA7, L)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            scores = []
            for singleKFold in allKFolds:
                x, f, d = trainLogisticRegression(singleKFold[1], singleKFold[0], l, 0.5)
                scores.append(getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_PCA7_kfold.append(cost)
            print("Lambda:", l, ", cost:", cost)
    plot.plotDCF(lambdas, minDCF_PCA7_kfold, "λ", "./logisticRegression/PCA7-kfold.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    
    # ----------------------------- PCA m = 6 - SINGLE FOLD -----------------------------
    minDCF_PCA6_singlefold = []
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(PCA6, L)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            x,f,d = trainLogisticRegression(DTR, LTR, l, 0.5)
            cost = cf.minimum_detection_costs(getScoresLogisticRegression(x, DEV), LEV, model[0], model[1], model[2])
            minDCF_PCA6_singlefold.append(cost)
            print("Lambda:", l, ", cost:", cost)
    plot.plotDCF(lambdas, minDCF_PCA6_singlefold, "λ", "./logisticRegression/PCA6-singlefold.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    
    # ----------------------------- PCA m = 6 - K FOLD -----------------------------
    minDCF_PCA6_kfold = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(PCA6, L)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            scores = []
            for singleKFold in allKFolds:
                x, f, d = trainLogisticRegression(singleKFold[1], singleKFold[0], l, 0.5)
                scores.append(getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_PCA6_kfold.append(cost)
            print("Lambda:", l, ", cost:", cost)
    plot.plotDCF(lambdas, minDCF_PCA6_kfold, "λ", "./logisticRegression/PCA6-kfold.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    
    return

def computeLogisticRegression(D, L, lambd = 1e-4):
    PCA6 = dr.PCA(D, L, 6)
    PCA7 = dr.PCA(D, L, 7)
    allKFoldsnoPCA, evaluationLabelsnoPCA = folds.Kfold_without_train(D, L)
    allKFoldsPCA7, evaluationLabelsPCA7 = folds.Kfold_without_train(PCA7, L)
    allKFoldsPCA6, evaluationLabelsPCA6 = folds.Kfold_without_train(PCA6, L)
    
    (DTR, LTR), (DEV, LEV) = folds.single_fold_without_train(D, L)
    (DTR7, LTR7), (DEV7, LEV7) = folds.single_fold_without_train(PCA7, L)
    (DTR6, LTR6), (DEV6, LEV6) = folds.single_fold_without_train(PCA6, L)
    
    # ------------------- NO PCA ---------------
    
    print("no PCA")    
    print("single-fold")
    for model in utils.models:
        for pi_T in utils.models:
            x, f, d = trainLogisticRegression(DTR, LTR, lambd, pi_T[0])
            minDCF = cf.minimum_detection_costs(getScoresLogisticRegression(x, DEV), LEV, model[0], model[1], model[2])  
            print("Lambda = ", lambd, "pi_T =", pi_T[0], "application with prior:", model[0], "minDCF = ", minDCF)
    
    print("K-fold")
    for model in utils.models: 
       
        for pi_T in utils.models:
            scores = []
            for singleKFold in allKFoldsnoPCA:
                x, f, d = trainLogisticRegression(singleKFold[1], singleKFold[0], lambd, pi_T[0])
                scores.append(getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            minDCF = cf.minimum_detection_costs(scores, evaluationLabelsnoPCA, model[0], model[1], model[2])
            
            print("Lambda = ", lambd, "pi_T =", pi_T[0], "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 7 ---------------
    
    print("PCA m = 7")
    print("single-fold")
    for model in utils.models:
        for pi_T in utils.models:
            x, f, d = trainLogisticRegression(DTR7, LTR7, lambd, pi_T[0])
            minDCF = cf.minimum_detection_costs(getScoresLogisticRegression(x, DEV7), LEV7, model[0], model[1], model[2])  
            print("Lambda = ", lambd, "pi_T =", pi_T[0], "application with prior:", model[0], "minDCF = ", minDCF)
    
    print("K-fold")
    for model in utils.models: 
        
        for pi_T in utils.models:
            scores = []
            for singleKFold in allKFoldsPCA7:
                x, f, d = trainLogisticRegression(singleKFold[1], singleKFold[0], lambd, pi_T[0])
                scores.append(getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA7, model[0], model[1], model[2])
            
            print("Lambda = ", lambd, "pi_T =", pi_T[0], "application with prior:", model[0], "minDCF = ", minDCF)
            
    # -------------- PCA m = 6 ---------------
    
    print("PCA m = 6")
    print("single-fold")
    for model in utils.models:
        for pi_T in utils.models:
            x, f, d = trainLogisticRegression(DTR6, LTR6, lambd, pi_T[0])
            minDCF = cf.minimum_detection_costs(getScoresLogisticRegression(x, DEV6), LEV6, model[0], model[1], model[2])  
            print("Lambda = ", lambd, "pi_T =", pi_T[0], "application with prior:", model[0], "minDCF = ", minDCF)
    
    print("K-fold")
    for model in utils.models: 
        
        for pi_T in utils.models:
            scores = []
            for singleKFold in allKFoldsPCA6:
                x, f, d = trainLogisticRegression(singleKFold[1], singleKFold[0], lambd, pi_T[0])
                scores.append(getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            minDCF = cf.minimum_detection_costs(scores, evaluationLabelsPCA6, model[0], model[1], model[2])
            
            print("Lambda = ", lambd, "pi_T =", pi_T[0], "application with prior:", model[0], "minDCF = ", minDCF)
            
            
    return
        
            
            

