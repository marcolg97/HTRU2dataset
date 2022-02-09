# -*- coding: utf-8 -*-
"""

@author: Giacomo Vitali and Marco La Gala

"""

import utils
import numpy as np
import logisticRegression as lr
import cost_functions as cf
import plot
import folds
import SVM
import GMM

def findBestLambdaEval (D, L, Dtest, Ltest):
    lambdas = np.logspace(-5, 5, num=50)
    
    
    minDCF_val = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            scores = []
            for singleKFold in allKFolds:
                x, f, d = lr.trainLogisticRegression(singleKFold[1], singleKFold[0], l, 0.5)
                scores.append(lr.getScoresLogisticRegression(x, singleKFold[2]))

            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_val.append(cost)
            print("Training Lambda:", l, ", cost:", cost)       
        
    minDCF_eval = []
    for model in utils.models: 
        print("Data for prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for l in lambdas:
            x,f,d = lr.trainLogisticRegression(D, L, l, 0.5)
            cost = cf.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
            minDCF_eval.append(cost)
            print("Test Lambda:", l, ", cost:", cost)
            
    
            
    plot.plotEvaluation(lambdas, minDCF_val, minDCF_eval, "λ", "./evaluationhp/logistic_regression.png", ticks=[10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5])
    return


def findBestCEval_unbalanced (D, L, Dtest, Ltest):
    k = 1.0
    rangeC=np.logspace(-5, -1, num=30)
    
    print("unbalanced SVM\n")
    minDCF_val = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for model in utils.models:
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for C in rangeC:
            scores = []
            
            for singleKFold in allKFolds:
                w = SVM.trainLinearSVM(singleKFold[1], singleKFold[0], C, k)
                scores.append(SVM.getScoresLinearSVM(w, singleKFold[2], k))

            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_val.append(cost)
            print("Training C:", C, ", cost:", cost)
            
    
    minDCF_eval = []    
    for model in utils.models: 
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for C in rangeC:
                            
            w = SVM.trainLinearSVM(D, L, C, k)
    
            cost = cf.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, k), Ltest, model[0], model[1], model[2])
            
            minDCF_eval.append(cost)
            print("Test C:", C, ", cost:", cost)
                
    plot.plotEvaluation(rangeC, minDCF_val, minDCF_eval, "C", "./evaluationhp/SVM_unbalanced.png")
    return

def findBestCEval_balanced (D, L, Dtest, Ltest, pi_T):
    print("balanced SVM", pi_T)
    
    k = 1.0
    rangeC=np.logspace(-5, -1, num=30)
    
    minDCF_eval = []
    
    for model in utils.models: 
                
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])
        
        for C in rangeC:
                            
            w = SVM.trainLinearSVM(D, L, C, k, pi_T)

            cost = cf.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, k), Ltest, model[0], model[1], model[2])
            
            minDCF_eval.append(cost)
            print("Test C:", C, ", cost:", cost)
    
    
    minDCF_val = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    for model in utils.models:
        
        print("\n prior", model[0], "cfp:", model[1], "cfn:", model[2])
        for C in rangeC:
            scores = []
            
            for singleKFold in allKFolds:
                w = SVM.trainLinearSVM(singleKFold[1], singleKFold[0], C, k, pi_T)
                scores.append(SVM.getScoresLinearSVM(w, singleKFold[2], k))

            scores=np.hstack(scores)
            cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
            minDCF_val.append(cost)
            print("Training C:", C, ", cost:", cost)


    filename = "./evaluationhp/SVM_balanced-" + str(pi_T) + ".png"
    plot.plotEvaluation(rangeC, minDCF_val, minDCF_eval, "C", filename)
    return
    
   
def findBestCompEval(D, L, Dtest, Ltest):
    
    print("FULL \n")
    full_eval = []
    model = utils.models[0]

    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        GMM0, GMM1 = GMM.trainGaussianClassifier(D, L, component)
        cost = cf.minimum_detection_costs(GMM.getScoresGaussianClassifier(Dtest, GMM0, GMM1) , Ltest, model[0], model[1], model[2])

        full_eval.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
    
    
    full_val = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainGaussianClassifier(singleKFold[1], singleKFold[0], component)
            scores.append(GMM.getScoresGaussianClassifier(singleKFold[2], GMM0, GMM1))
        
        scores=np.hstack(scores)
        cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
        
        full_val.append(cost)
        print("component:", 2**(component), "cost:", cost)
            
    plot.histrogram(full_val, full_eval, "./evaluationhp/GMM_fullcov.png")
    
    
    
    
    print("DIAG \n")
    diag_eval = []
    
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        GMM0, GMM1 = GMM.trainNaiveBayes(D, L, component)
        cost = cf.minimum_detection_costs(GMM.getScoresNaiveBayes(Dtest, GMM0, GMM1) , Ltest, model[0], model[1], model[2])

        diag_eval.append(cost)
        print("component:", 2**(component), "cost:", cost)
    


    diag_val = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    for component in range(7):
        
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainNaiveBayes(singleKFold[1], singleKFold[0], component)
            scores.append(GMM.getScoresNaiveBayes(singleKFold[2], GMM0, GMM1))
        
        scores=np.hstack(scores)
        cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
        
        diag_val.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
   
    plot.histrogram(diag_val, diag_eval, "./evaluationhp/GMM_diagcov.png")
    
    
    
    print("TIED \n")
    tied_eval = []
     
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        GMM0, GMM1 = GMM.trainTiedCov(D, L, component)
        cost = cf.minimum_detection_costs(GMM.getScoresTiedCov(Dtest, GMM0, GMM1) , Ltest, model[0], model[1], model[2])

        tied_eval.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
   
   
    tied_val = []
    allKFolds, evaluationLabels = folds.Kfold_without_train(D, L)
    
    print("\nStarting draw with prior", model[0], "cfp:", model[1], "cfn:", model[2])
    
    for component in range(7):
        scores = []
        for singleKFold in allKFolds:
            GMM0, GMM1 = GMM.trainTiedCov(singleKFold[1], singleKFold[0], component)
            scores.append(GMM.getScoresTiedCov(singleKFold[2], GMM0, GMM1))
        
        scores=np.hstack(scores)
        cost = cf.minimum_detection_costs(scores, evaluationLabels, model[0], model[1], model[2])
        
        tied_val.append(cost)
        print("component:", 2**(component), "cost:", cost)
    
    plot.histrogram(tied_val, tied_eval, "./evaluationhp/GMM_tiedcov.png")
    
    return

def EvaluateHyperParameterChosen(D, L, Dtest, Ltest):
    findBestLambdaEval(D, L, Dtest, Ltest)
    findBestCEval_unbalanced(D, L, Dtest, Ltest)
    
    findBestCEval_balanced(D, L, Dtest, Ltest, 0.5)
    findBestCEval_balanced(D, L, Dtest, Ltest, 0.1)
    findBestCEval_balanced(D, L, Dtest, Ltest, 0.9)
    
    findBestCompEval(D, L, Dtest, Ltest)

    
    return