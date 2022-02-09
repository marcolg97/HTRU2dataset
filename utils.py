# -*- coding: utf-8 -*-
"""

@authors: Giacomo Vitali and Marco La Gala

"""

# -------------------- CONSTANTS ---------------------

classesNames = ["False pulsar signal", "True pulsar signal"]
featuresNames = ["Mean of the integrated profile",
                 "Standard deviation of the integrated profile",
                 "Excess kurtosis of the integrated profile",
                 "Skewness of the integrated profile",
                 "Mean of the DM-SNR curve",
                 "Standard deviation of the DM-SNR curve",
                 "Excess kurtosis of the DM-SNR curve",
                 "Skewness of the DM-SNR curve"]

# Here we have the models with [pi, cfp, cfn]
models = [ [0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1] ]

# -------------------- FUNCTIONS ---------------------

def mcol(v):
    # Function to transform 1-dim vectors to column vectors.
    return v.reshape((v.size, 1))

def mrow(v):
    # Function to transform 1-dim vecotrs to row vectors.
    return (v.reshape(1, v.size))