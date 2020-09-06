# -*- coding: utf-8 -*-
""" settings """

import os


###############################################################################
#                                   Dataset                                   #
###############################################################################
GENERATED_DATA_DIRECTORY = "tmp"
TRAINING_DATA_DIRECTORY = "trainingdata"
FULL_DATASET_PATH = os.path.join(
    TRAINING_DATA_DIRECTORY,
    'spatialpyramidfeatures4caltech101/spatialpyramidfeatures4caltech101.mat'
)

PERSON_NUMBER = 102  # person number for evaluation

###############################################################################
#                       Incremental Dictionary Learning                       #
###############################################################################

N_TRAINSAMPLES = 30  # training samples per class/category

PARS = dict(
    gamma=1e-6,
    lambda_=0.5,
    mu=0.6,  # ||Q-AX||^2
    nu1=1e-6,  # regularization of A
    nu2=1e-6,  # regularization of W
    rho=10,  # initial learning rate
    maxIters=20,  # iteration number for incremental learning
    batchSize=60,
    iterationini=5,  # iteration number for initialization
    ntrainsamples=N_TRAINSAMPLES,
    numBases=PERSON_NUMBER*N_TRAINSAMPLES,  # dictionary size
    dataset='caltech101',
)

###############################################################################
#                                   LC-KSVD2                                  #
###############################################################################

SPARSITYTHRES = 40  # sparsity prior
SQRT_ALPHA = 0.0012  # weights for label constraint term # not used
SQRT_BETA = 0.0012  # weights for classification err term # not used
ITERATIONS = 50  # iteration number  # not used
ITERATIONS4INI = 20  # iteration number for initialization # not used
DICTSIZE = PERSON_NUMBER*N_TRAINSAMPLES  # dictionary size  # not used and repeated at pars['numBases']
