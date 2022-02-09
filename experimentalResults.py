# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""

import dimensionality_reduction as dr
import gaussianClassifier as gc
import logisticRegression as lr
import SVM
import GMM
import utils
import cost_functions as cf

def ER_MVG(D,L, Dtest, Ltest):
    
    print("MVG Full-Cov")
    mean0, sigma0, mean1, sigma1 = gc.trainGaussianClassifier(D, L)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest), Ltest, model[0], model[1], model[2])
        print("prior:", model[0], "minDCF:", minDCF_test)
    
    print("MVG Diag-Cov")
    mean0, sigma0, mean1, sigma1 = gc.trainNaiveBayes(D, L)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest), Ltest, model[0], model[1], model[2])
        print("prior:", model[0], "minDCF:", minDCF_test)
    
    print("MVG Tied-Cov")
    mean0, sigma0, mean1, sigma1 = gc.trainTiedCov(D, L)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(gc.getScoresGaussianClassifier(mean0, sigma0, mean1, sigma1, Dtest), Ltest, model[0], model[1], model[2])
        print("prior:", model[0], "minDCF:", minDCF_test)
    
    return

def ER_LR(D, L, Dtest, Ltest):
    
    lambd = 10**(-4)
    prior = 0.5
    
    print("Logistic Regression pi_T = 0.5")
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
        print("Lambda:", lambd, "pi_T", prior, "prior:", model[0], "minDCF:", minDCF_test) 
        
    prior = 0.1
    print("Logistic Regression pi_T = 0.1")
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
        print("Lambda:", lambd, "pi_T", prior, "prior:", model[0], "minDCF:", minDCF_test)
        
    prior = 0.9
    print("Logistic Regression pi_T = 0.9")
    x, f, d = lr.trainLogisticRegression(D, L, lambd, prior)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(lr.getScoresLogisticRegression(x, Dtest), Ltest, model[0], model[1], model[2])
        print("Lambda:", lambd, "pi_T", prior, "prior:", model[0], "minDCF:", minDCF_test)
    
    return

def ER_SVM(D, L, Dtest, Ltest):
    
    K_linear = 1.0
       
    
    C_linear = 10**(-2)
    print("Linear SVM unbalanced")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
          
    C_linear = 10**(-3)
    prior = 0.5
    print("Linear SVM with pi_T = 0.5")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear, pi_T = prior)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
    
    C_linear = 6*10**(-3)
    prior = 0.1
    print("Linear SVM with pi_T = 0.1")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear, pi_T = prior)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
     
    C_linear = 7*10**(-4)
    prior = 0.9
    print("Linear SVM with pi_T = 0.9")
    w = SVM.trainLinearSVM(D, L, C = C_linear, K = K_linear, pi_T = prior)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(SVM.getScoresLinearSVM(w, Dtest, K = K_linear), Ltest, model[0], model[1], model[2])
        print("C:", C_linear, "prior:", model[0], "minDCF:", minDCF_test)
          
    C_quadratic = 5*10**(-5)
    c = 15
    d = 2
    K_quadratic = 1.0
    print("Quadratic SVM")
    x = SVM.trainPolynomialSVM(D, L, C = C_quadratic, K = K_quadratic, c = c, d = d)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(SVM.getScoresPolynomialSVM(x, D, L, Dtest, K = K_quadratic, c = c, d = d), Ltest, model[0], model[1], model[2])
        print("C:", C_quadratic, "c:", c, "d:", d, "prior:", model[0], "minDCF:", minDCF_test)
     
    C_RBF = 10**(-1)
    gamma = 10**(-3)
    K_RBF = 1.0
    print("RBF Kernel SVM")
    x = SVM.trainRBFKernel(D, L, gamma = gamma, K = K_RBF, C = C_RBF)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(SVM.getScoresRBFKernel(x, D, L, Dtest, gamma = gamma, K = K_RBF), Ltest, model[0], model[1], model[2])
        print("C:", C_RBF, "gamma:", gamma, "prior:", model[0], "minDCF:", minDCF_test)

    return

def ER_GMM(D, L, Dtest, Ltest):
    
    nComp_full = 4 # 2^4 = 16
    nComp_diag = 5 # 2^5 = 32
    nComp_tied = 6 # 2^6 = 64
    
    
    print("GMM Full-Cov", 2**(nComp_full), "components")
    GMM0, GMM1 = GMM.trainGaussianClassifier(D, L, nComp_full)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(GMM.getScoresGaussianClassifier(Dtest, GMM0, GMM1), Ltest, model[0], model[1], model[2])
        print("components:", 2**(nComp_full), "prior:", model[0], "minDCF:", minDCF_test)
        

    print("GMM Diag-Cov", 2**(nComp_diag), "components")
    GMM0, GMM1 = GMM.trainNaiveBayes(D, L, nComp_diag)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(GMM.getScoresNaiveBayes(Dtest, GMM0, GMM1), Ltest, model[0], model[1], model[2])
        print("components:", 2**(nComp_diag), "prior:", model[0], "minDCF:", minDCF_test)
        
    
    print("GMM Tied-Cov", 2**(nComp_tied), "components")
    GMM0, GMM1 = GMM.trainTiedCov(D, L, nComp_tied)
    
    for model in utils.models:
        minDCF_test = cf.minimum_detection_costs(GMM.getScoresTiedCov(Dtest, GMM0, GMM1), Ltest, model[0], model[1], model[2])
        print("components:", 2**(nComp_tied), "prior:", model[0], "minDCF:", minDCF_test)
    
    return

def computeExperimentalResults(D, L, Dtest, Ltest):
    
    PCA7 = dr.PCA(D, L, 7)
    PCA7_test = dr.PCA(Dtest, Ltest, 7)
    
    # no PCA
    print("no PCA")
    ER_MVG(D, L, Dtest, Ltest)
    ER_LR(D, L, Dtest, Ltest)
    ER_SVM(D, L, Dtest, Ltest)
    ER_GMM(D, L, Dtest, Ltest)
    
    # PCA m = 7
    print("PCA m = 7")
    ER_MVG(PCA7, L, PCA7_test, Ltest)
    ER_LR(PCA7, L, PCA7_test, Ltest)
    ER_SVM(PCA7, L, PCA7_test, Ltest)
    ER_GMM(PCA7, L, PCA7_test, Ltest)
    
    return