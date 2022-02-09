# -*- coding: utf-8 -*-
"""

@author: Giacomo Vitali and Marco La Gala

"""

import numpy as np

def single_fold(D, L, trainModel, getScores, seed=0):
    # Get an integer nTrain representing 2/3 of the dataset dimension
    nTrain = int(D.shape[1]*2.0/3.0)
    # Generate a random seed
    np.random.seed(seed)
    # D.shape[1] is an integer, so according to numpy documentation the
    # permutation function will work on the vector produced by arange(D.shape[1]),
    # which will be a 1-dim vector of 8929 elements with evenly spaced values
    # between 0 and 8928. Then these values are permuted.
    # The shuffle isn't really random, in fact with the same seed and the
    # same integer passed as parameter to the permutation function the
    # result will always be the same. This is because there's an algorithm
    # that outputs the values based on the seed and it obviously has a
    # deterministic behavior, so now we don't have to worry about getting
    # different values from the ones proposed in the pdf because they will
    # be the same.
    idx = np.random.permutation(D.shape[1])
    # In idxTrain we select only the first elements of idx
    idxTrain = idx[0:nTrain]
    # In idxEval we select only the last elements of idx
    idxEval = idx[nTrain:]
    # The Data matrix for TRaining need to be reduced to a 8xm matrix.
    # At the same time, by passing the random vector idxTrain as a parameter
    # for the slice, we will actually select only the feature vectors (columns)
    # at indexes specified in the idxTrain vector.
    DTR = D[:, idxTrain]
    # The Data matrix for EValuation need to be reduced to a 8xn matrix.
    # Same process of DTR. THERE CAN'T BE OVERLAPPING FEATURE VECTORS BETWEEN
    # DTR AND DEV, SINCE THE VALUES OF IDX THAT HAVE BEEN USED AS INDEXES ARE
    # ALL DIFFERENT.
    DEV = D[:, idxEval]
    # The Label vector for TRaining need to be reduced to size of training samples.
    # Same process, we pass an array with indexes.
    LTR = L[idxTrain]
    # The Label vector for EValuation need to be reduced to size of evaluation samples.
    # Same process, we pass an array with indexes.
    LEV = L[idxEval]
    
    
    # Now we train the model and calculate the score on the evaluation set
    mean0, sigma0, mean1, sigma1 = trainModel(DTR, LTR)
    scores = getScores(mean0, sigma0, mean1, sigma1, DEV)
    
    return scores, LEV

def Kfold(D, L, trainModel, getScores, K=5, seed=0):
    # 1. Split the dataset in k folds, we choose 5
    foldSize = int(D.shape[1]/K)
       
    folds = []
    labels = []
    
    # Generate a random seed and compute a permutation from 0 to 8929
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    
    for i in range(K):
        fold_idx = idx[(i*foldSize) : ((i+1)*(foldSize))]
        folds.append(D[:,fold_idx])
        labels.append(L[fold_idx])
        
    
    # 2. Re-arrange folds and train the model
    scores = [] # Each iteration we save here the score for the evaluation set. At the end we have the score of the whole dataset
    evaluationLabels = [] # Each iteration we save here the labels for the evaluation set. At the end we have the labels in the new shuffled order 

    # The algorithm is trained and evaluated K times
    for i in range(K):
        # Each time we iterate, we create K-1 training folds and one evaluation set to train and evaluate the model
        trainingFolds = []
        trainingLabels = []
        evaluationFold = []
       
        
        # We iterate over the folds. The i-th fold will be the evaluation one
        for j in range(K):
            if j==i:
                evaluationFold.append(folds[i])
                evaluationLabels.append(labels[i])
            else:
                trainingFolds.append(folds[j])
                trainingLabels.append(labels[j])

        trainingFolds=np.hstack(trainingFolds)
        trainingLabels=np.hstack(trainingLabels)
        evaluationFold = np.hstack(evaluationFold)
        
        # Now we train the model and calculate the score on the evaluation set
        mean0, sigma0, mean1, sigma1 = trainModel(trainingFolds, trainingLabels)
        scores.append(getScores(mean0, sigma0, mean1, sigma1, evaluationFold))
        
    scores=np.hstack(scores)
    evaluationLabels=np.hstack(evaluationLabels)
    
    return scores, evaluationLabels

def single_fold_without_train(D, L, seed=0):
    # Get an integer nTrain representing 2/3 of the dataset dimension
    nTrain = int(D.shape[1]*2.0/3.0)
    # Generate a random seed
    np.random.seed(seed)
    # D.shape[1] is an integer, so according to numpy documentation the
    # permutation function will work on the vector produced by arange(D.shape[1]),
    # which will be a 1-dim vector of 8929 elements with evenly spaced values
    # between 0 and 8928. Then these values are permuted.
    # The shuffle isn't really random, in fact with the same seed and the
    # same integer passed as parameter to the permutation function the
    # result will always be the same. This is because there's an algorithm
    # that outputs the values based on the seed and it obviously has a
    # deterministic behavior, so now we don't have to worry about getting
    # different values from the ones proposed in the pdf because they will
    # be the same.
    idx = np.random.permutation(D.shape[1])
    # In idxTrain we select only the first elements of idx
    idxTrain = idx[0:nTrain]
    # In idxEval we select only the last elements of idx
    idxEval = idx[nTrain:]
    # The Data matrix for TRaining need to be reduced to a 8xm matrix.
    # At the same time, by passing the random vector idxTrain as a parameter
    # for the slice, we will actually select only the feature vectors (columns)
    # at indexes specified in the idxTrain vector.
    DTR = D[:, idxTrain]
    # The Data matrix for EValuation need to be reduced to a 8xn matrix.
    # Same process of DTR. THERE CAN'T BE OVERLAPPING FEATURE VECTORS BETWEEN
    # DTR AND DEV, SINCE THE VALUES OF IDX THAT HAVE BEEN USED AS INDEXES ARE
    # ALL DIFFERENT.
    DEV = D[:, idxEval]
    # The Label vector for TRaining need to be reduced to size of training samples.
    # Same process, we pass an array with indexes.
    LTR = L[idxTrain]
    # The Label vector for EValuation need to be reduced to size of evaluation samples.
    # Same process, we pass an array with indexes.
    LEV = L[idxEval]
            
    return (DTR, LTR) , (DEV, LEV)


def Kfold_without_train(D, L, seed=0, K=5):
    # 1. Split the dataset in k folds, we choose 5
    foldSize = int(D.shape[1]/K)
       
    folds = []
    labels = []
    
    # Generate a random seed and compute a permutation from 0 to 8929
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    
    for i in range(K):
        fold_idx = idx[(i*foldSize) : ((i+1)*(foldSize))]
        folds.append(D[:,fold_idx])
        labels.append(L[fold_idx])
        
    
    # 2. Re-arrange folds and train the model
    evaluationLabels = [] # Each iteration we save here the labels for the evaluation set. At the end we have the labels in the new shuffled order 


    allKFolds = []
    
    # The algorithm is trained and evaluated K times
    for i in range(K):
        # Each time we iterate, we create K-1 training folds and one evaluation set to train and evaluate the model
        trainingFolds = []
        trainingLabels = []
        evaluationFold = []
       
        
        # We iterate over the folds. The i-th fold will be the evaluation one
        for j in range(K):
            if j==i:
                evaluationFold.append(folds[i])
                evaluationLabels.append(labels[i])
            else:
                trainingFolds.append(folds[j])
                trainingLabels.append(labels[j])

        trainingFolds=np.hstack(trainingFolds)
        trainingLabels=np.hstack(trainingLabels)
        evaluationFold = np.hstack(evaluationFold)
    
        singleKFold = [trainingLabels, trainingFolds, evaluationFold]
        allKFolds.append(singleKFold)
        
        
    evaluationLabels=np.hstack(evaluationLabels)
        
    return allKFolds, evaluationLabels 