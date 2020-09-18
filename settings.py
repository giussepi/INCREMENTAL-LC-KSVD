# -*- coding: utf-8 -*-
""" settings """

import os


###############################################################################
#                                   General                                    #
###############################################################################

GENERATED_DATA_DIRECTORY = "tmp"


###############################################################################
#                                   Dataset                                   #
###############################################################################

DATASETS_DIRECTORY = "trainingdata"

# # spatialpyramidfeatures4caltech101 ###########################################
# FULL_DATASET_PATH = os.path.join(
#     DATASETS_DIRECTORY,
#     'spatialpyramidfeatures4caltech101'.
#     'spatialpyramidfeatures4caltech101.mat'
# )

# CLASS_NUMBER = 102  # class number for evaluation. Person number for caltech101

# PatchCamelyon ###############################################################
# my_raw_dataset_test.json
TRAINING_DATA_DIRECTORY_DATASET_PATH = os.path.join(
    DATASETS_DIRECTORY,
    'patchcamelyon',
    'my_raw_dataset_train.json'
)
TESTING_DATA_DIRECTORY_DATASET_PATH = os.path.join(
    DATASETS_DIRECTORY,
    'patchcamelyon',
    'my_raw_dataset_test.json'
)
CLASS_NUMBER = 2  # class number for evaluation. Contains tumour tissue or not.


###############################################################################
#                       Incremental Dictionary Learning                       #
###############################################################################

N_TRAINSAMPLES = 15  # 30  # training samples per class/category

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
    ntrainsamples=N_TRAINSAMPLES,  # only for spatialpyramidfeatures4caltech101 dataset
    numBases=CLASS_NUMBER*N_TRAINSAMPLES,  # dictionary size
    dataset='patchcamelyon',  # 'caltech101',
)

###############################################################################
#                                   LC-KSVD2                                  #
###############################################################################

SPARSITYTHRES = 15  # 40  # sparsity prior
SQRT_ALPHA = 0.0012  # weights for label constraint term # not used
SQRT_BETA = 0.0012  # weights for classification err term # not used
ITERATIONS = 50  # iteration number  # not used
ITERATIONS4INI = 20  # iteration number for initialization # not used
DICTSIZE = CLASS_NUMBER*N_TRAINSAMPLES  # dictionary size  # not used and repeated at pars['numBases']
