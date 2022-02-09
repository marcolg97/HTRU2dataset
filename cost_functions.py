# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""

import numpy as np

def createConfusionMatrix(pl, LEV, K):
    # Matrix with all zeros KxK
    matrix = np.zeros((K, K)).astype(int)
   
    for i in range(LEV.size):
        # Update "counter" in proper position
        matrix[pl[i], LEV[i]] += 1
    return matrix


def evaluationBinaryTask(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    # Compute empirical Bayes risk, that is the cost that we pay due to our
    # decisions c* for the test data.
    DCFu = pi1*Cfn*FNR+(1-pi1)*Cfp*FPR
    return DCFu

def normalizedEvaluationBinaryTask(pi1, Cfn, Cfp, DCFu):
    # Define vector with dummy costs
    dummyCosts = np.array([pi1*Cfn, (1-pi1)*Cfp])
    # Compute risk for an optimal dummy system
    index = np.argmin(dummyCosts)
    # Compute normalized DCF
    DCFn = DCFu/dummyCosts[index]
    return DCFn


def minimum_detection_costs (llr, labels, pi1, cfn, cfp):
    
    # -------------------- MINIMUM DETECTION COSTS ----------------------------
    # We can compute the optimal threshold for a given application on the same
    # validation set that we're analyzing, and use such threshold for the test
    # population (K-fold cross validation can be also exploited to extract
    # validation sets from the training data when validation data is not available).
    # We can compute the normalized DCF over the test set using all possible
    # thresholds, and select its minimum value. This represents a lower bound
    # for the DCF that our system can achieve (minimum DCF).
    # To compute the minimum cost, we consider a set of threshold corresponding
    # to the set of test scores (llr), sorted in increasing order. For each
    # threshold of this set, we compute the confusion matrix on the test set
    # itself and the corresponding normalized DCF using the code developed in
    # the previous section. The minimum DCF is the minimum of the obtained values.
    testScoresSorted = np.sort(llr)
    # Define empty lists to store DCFs for the applications
    
    
    normalizedDCF= []
    
    for t in testScoresSorted:
                
        # Now, if the llr is > than the threshold => predicted class is 1
        # If the llr is <= than the threshold => predicted class is 0
        predictedLabels = (llr > t).astype(int)
        # Compute the confusion matrix
        confusionMatrix = createConfusionMatrix(predictedLabels, labels, 2)
        
        DCFu = evaluationBinaryTask(pi1, cfn, cfp, confusionMatrix)
        
        normalizedDCF.append(normalizedEvaluationBinaryTask(pi1, cfn, cfp, DCFu))
        
    index = np.argmin(normalizedDCF)
        
    return normalizedDCF[index]

def compute_actual_DCF(llr, labels, pi1, cfn, cfp):
    
    predictions = (llr > ( -np.log(pi1/(1 - pi1)))).astype(int)
    
    confusionMatrix = createConfusionMatrix(predictions, labels, 2)
    DCFu = evaluationBinaryTask(pi1, cfn, cfp, confusionMatrix)
        
    normalizedDCF = normalizedEvaluationBinaryTask(pi1, cfn, cfp, DCFu)
        
    return normalizedDCF

def computeOptimalBayesDecisionBinaryTaskTHRESHOLD(llrs, labels, t):
    # Now, if the llr is > than the threshold => predicted class is 1
    # If the llr is <= than the threshold => predicted class is 0
    predictions = (llrs > t).astype(int)
    # Compute the confusion matrix
    m = createConfusionMatrix(predictions, labels, 2)
    return m

def computeFPRTPR(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    TPR = 1-FNR
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    return (FPR, TPR)