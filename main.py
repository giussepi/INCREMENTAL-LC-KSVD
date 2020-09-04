# -*- coding: utf-8 -*-
""" main """

import json
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from spams import trainDL, lasso, omp
from scipy.io import loadmat

generated_data_directory = "tmp"
training_data_directory = "trainingdata"

file_path = os.path.join(
    training_data_directory,
    'spatialpyramidfeatures4caltech101/spatialpyramidfeatures4caltech101.mat'
)
data = loadmat(file_path)
# data['filenameMat'].shape (1, 102)
# data['featureMat'].shape (3000, 9144)
# data['labelMat'].shape (102, 9144)

# constant
personnumber = 102  # person number for evaluation
# constant for incremental dictionary learning


ntrainsamples = 30

pars = dict(
    gamma=1e-6,
    lambda_=0.5,
    mu=0.6,  # ||Q-AX||^2
    nu1=1e-6,  # regularization of A
    nu2=1e-6,  # regularization of W
    rho=10,  # initial learning rate
    maxIters=20,  # iteration number for incremental learning
    batchSize=60,
    iterationini=5,  # iteration number for initialization
    ntrainsamples=ntrainsamples,
    numBases=personnumber*ntrainsamples,  # dictionary size
    dataset='caltech101',
)

# constant for LC-KSVD2
sparsitythres = 40  # sparsity prior
sqrt_alpha = 0.0012  # weights for label constraint term
sqrt_beta = 0.0012  # weights for classification err term
iterations = 50  # iteration number
iterations4ini = 20  # iteration number for initialization
dictsize = personnumber*30  # dictionary size


def obtaintraingtestingsamples(featureMatrix, labelMatrix, numPerClass):
    """
    Obtain training and testing features by random sampling

    Args:
        featureMatrix      - input features
        labelMatrix        - label matrix for input features
        numPerClass        - number of training samples from each category

    Return:
        testPart           - testing features
        HtestPart          - label matrix for testing features
        trainPart          - training features
        HtrainPart         - label matrix for training features
    """
    numClass = labelMatrix.shape[0]  # number of objects
    testPart = np.empty((featureMatrix.shape[0], 0))
    HtestPart = np.empty((labelMatrix.shape[0], 0))
    trainPart = np.empty((featureMatrix.shape[0], 0))
    HtrainPart = np.empty((labelMatrix.shape[0], 0))

    for classid in range(numClass):
        # col_ids = find(labelMatrix(classid, : ) == 1)
        col_ids = np.array(np.nonzero(labelMatrix[classid, :] == 1)).ravel()

        # data_ids = find(colnorms_squared_new(featureMatrix(: , col_ids)) > 1e-6)
        data_ids = np.array(np.nonzero(np.sum(featureMatrix[:, col_ids]**2, axis=0) > 1e-6)).ravel()
        # % ensure no zero data elements
        # perm = randperm(length(data_ids))
        # perm = np.random.permutation(data_ids.shape[0])
        # # perm = [1:length(data_ids)];

        # trainids = col_ids(data_ids(perm(1: numPerClass)))
        # trainids = col_ids[data_ids[perm[:numPerClass]]]
        trainids = col_ids[np.random.choice(data_ids, numPerClass, replace=False)]
        # testids = setdiff(col_ids, trainids)
        testids = np.setdiff1d(col_ids, trainids)

        # testPart = [testPart featureMatrix(:, testids)]
        testPart = np.c_[testPart, featureMatrix[:, testids]]
        # HtestPart = [HtestPart labelMatrix(:, testids)]
        HtestPart = np.c_[HtestPart, labelMatrix[:, testids]]
        # trainPart = [trainPart featureMatrix(:, trainids)]
        trainPart = np.c_[trainPart, featureMatrix[:, trainids]]
        # HtrainPart = [HtrainPart labelMatrix(:, trainids)]
        HtrainPart = np.c_[HtrainPart, labelMatrix[:, trainids]]

    return testPart, HtestPart, trainPart, HtrainPart


# get training and testing data
testing_feats, H_test, training_feats, H_train = obtaintraingtestingsamples(
    data['featureMat'], data['labelMat'], pars['ntrainsamples'])
# testing_feats (3000, 6084)
# H_test (102, 6084)
# training_feats (3000, 3060)
# H_train (102, 3060)

# get the subsets of training data and testing data
# it is related to the variable 'personnumber'
# [labelvector_train,~] = find(H_train);
labelvector_train, _ = H_train.nonzero()  # 3060
# [labelvector_test, ~] = find(H_test)
labelvector_test, _ = H_test.nonzero()  # 6084
# trainsampleid = find(labelvector_train <= personnumber)
trainsampleid = np.nonzero(labelvector_train <= personnumber)[0]  # 3060
# testsampleid = find(labelvector_test <= personnumber)
testsampleid = np.nonzero(labelvector_test <= personnumber)[0]  # 6084
# trainingsubset = training_feats(:, trainsampleid)
trainingsubset = training_feats[:, trainsampleid]  # 3000, 3060
# testingsubset = testing_feats(:, testsampleid)
testingsubset = testing_feats[:, testsampleid]  # 3000, 6084
# H_train_subset = H_train(1: personnumber, trainsampleid)
H_train_subset = H_train[: personnumber+1, trainsampleid]  # 102, 3060
# H_test_subset = H_test(1:personnumber,testsampleid);
H_test_subset = H_test[: personnumber+1, testsampleid]  # 102, 6084

###############################################################################
#                     Verified with matlabscript till here                    #
###############################################################################

##########
# Incremental Dictionary learning


def paramterinitialization(training_feats, H_train, para):
    """ paramter initialization for incremental dictionary learning """
    dictsize = para.get('numBases')
    iterations = para.get('iterationini')
    numClass = H_train.shape[0]  # number of objects
    Dinit = np.empty((training_feats.shape[0], 0))  # for C-Ksvd and D-Ksvd
    dictLabel = np.empty((numClass, 0), dtype=np.int)
    numPerClass = dictsize//numClass
    param1 = {
        'mode': 2,
        'K': para.get('numBases'),  # size of the dictionary
        'lambda1': para.get('lambda_'),
        'lambda2': 0,
        'iter': iterations
    }
    param2 = {
        'lambda1': para.get('lambda_'),
        'lambda2': 0,
        'mode': 2
    }

    for classid in range(numClass):
        labelvector = np.zeros((numClass, 1), dtype=np.int)  # eyes
        labelvector[classid] = 1
        # dictLabel = [dictLabel repmat(labelvector, 1, numPerClass)]
        dictLabel = np.c_[dictLabel, np.tile(labelvector, (1, numPerClass))]

    for classid in range(numClass):
        col_ids = np.array(np.nonzero(H_train[classid, :] == 1)).ravel()
        # ensure no zero data elements are chosen
        data_ids = np.array(np.nonzero(np.sum(training_feats[:, col_ids]**2, axis=0) > 1e-6)).ravel()
        # perm = randperm(length(data_ids));
        # perm = [1:length(data_ids)];

        # Initilization for LC-KSVD (perform KSVD in each class)
        # Dpart = training_feats(: , col_ids(data_ids(perm(1: numPerClass))))
        Dpart = training_feats[:, col_ids[np.random.choice(data_ids, numPerClass, replace=False)]]
        param1['D'] = Dpart  # initial dictionary
        Dpart = trainDL(training_feats[:, col_ids[data_ids]], **param1)
        Dinit = np.c_[Dinit, Dpart]
        labelvector = np.zeros((numClass, 1))  # eyes
        labelvector[classid] = 1
        # dictLabel = [dictLabel repmat(labelvector,1,numPerClass)];
        dictLabel = np.c_[dictLabel, np.tile(labelvector, (1, numPerClass))]

    param1['D'] = np.asfortranarray(Dinit)  # initial dictionary
    # RuntimeError: matrix arg 10 must be a 2d double Fortran Array
    Dinit = trainDL(training_feats, **param1)

    Xinit = lasso(training_feats, Dinit, **param2)

    # learning linear classifier parameters
    tmp = np.linalg.inv(Xinit@Xinit.T+np.eye(*(Xinit@Xinit.T).shape))@Xinit
    Winit = tmp@H_train.T
    Winit = Winit.T

    Q = np.zeros((dictsize, training_feats.shape[1]))  # energy matrix

    for frameid in range(training_feats.shape[1]):
        label_training = H_train[:, frameid]
        maxid1 = label_training.argmax(0)

        for itemid in range(Dinit.shape[1]):
            label_item = dictLabel[:, itemid]
            maxid2 = label_item.argmax(0)

            if maxid1 == maxid2:
                Q[itemid, frameid] = 1
            # else:
            #     Q[itemid, frameid] = 0

    Tinit = tmp@Q.T
    Tinit = Tinit.T

    return Dinit, Winit, Tinit, Q


# initialization
# uncomment
# [Dinit, Winit, Tinit, Q_train] = paramterinitialization(trainingsubset, H_train_subset, pars)

###############################################################################
#                          working good                                       #
###############################################################################

# uncomment
# pars['D'] = Dinit
# pars['A'] = Tinit
# pars['W'] = Winit

print('\nIncremental dictionary learning...')


def getobjective_lc(D, S, Y, W, H, A, Q, lambda_, mu):
    """  """
    E1 = D.astype(np.float64)@S - Y.astype(np.float64)
    E2 = A.astype(np.float64)@S - Q.astype(np.float64)
    E3 = W.astype(np.float64)@S - H.astype(np.float64)

    fresidue1 = np.sum(E1**2)  # reconstruction error
    fresidue2 = np.sum(E2**2)  # optimal sparse code error
    # TODO: REview why this happens, I believe it's an issue related with W.
    # maybe it's calculated in a lazy way I'm not sure.....
    # numpy.linalg.LinAlgError: Last 2 dimensions of the array must be square
    # fresidue3 = np.sum(E3**2)  # classification error
    fresidue3 = np.sum(np.multiply(E3, E3))  # classification error
    freg = lambda_*np.sum(np.absolute(S))

    # fobj = 0.5*(fresidue1 + mu1*fresidue2 + mu2*fresidue3) + freg;
    # fobj = 0.5*(fresidue1 + mu1*fresidue2 + mu2*fresidue3);
    fobj = 0.5*(mu*fresidue2 + (1-mu)*fresidue3)

    return fobj, fresidue1, fresidue2, fresidue3, freg


def onlineDictionaryLearning(pars, trainingdata, HMat, QMat):
    """
    online dictionary learning using stochastic gradient descent algorithm
    Args:
        pars: learning paramters
           'D': initial dictionary
           'A': initial transform matrix
           'W': initial classifier
           'mu': reguliarization parameter
           'nu1': reguliarization parameter
           'nu2': reguliarization parameter
           'maxIters': iteration number
           'rho': learning rate parameter
        trainingdata: input training des. with size of n X N
        HMat: label matrix of training des. with size of m X N
        QMat: optimal code matrix of training des. with size of K X N

    Returns:
        model: learned paramters
            'D': learned dictionary
            'A': learned transform matrix
            'W': learned classifier
        fobj_avg: average objective function value

    Author: Zhuolin Jiang (zhuolin@umiacs.umd.edu)
    """
    num_images = trainingdata.shape[1]
    num_bases = pars['numBases']
    num_iters = pars['maxIters']
    gamma = pars['gamma']  # sparse coding parameters
    lambda_ = pars['lambda_']
    nu1 = pars['nu1']
    nu2 = pars['nu2']
    mu = pars['mu']
    rho = pars['rho']
    # n0 = num_images/10
    n0 = num_images/(pars['batchSize']*10)
    model = dict(
        D=pars['D'],  # dictionary
        W=pars['W'],  # classifier
        A=pars['A'],  # transform matrix
    )

    param = {
        'lambda1': pars['lambda_'],
        'lambda2': 0,
        'mode': 2
    }

    # crf iterations
    fobj_avg = dict()

    if not os.path.isdir(generated_data_directory):
        os.mkdir(generated_data_directory)

    for iter_ in range(num_iters):
        tic = time.time()
        # Take a random permutation of the samples
        filename = 'permute_{}_{}_{}.npy'.format(iter_, num_bases, pars['dataset'])
        full_path = os.path.join(generated_data_directory, filename)

        if os.path.isfile(full_path):
            ind_rnd = np.load(full_path, allow_pickle=False, fix_imports=False)
        else:
            ind_rnd = np.random.permutation(num_images)
            np.save(full_path, ind_rnd, allow_pickle=False, fix_imports=False)

        for batch in range(num_images//pars['batchSize']):
            # load the dataset
            # we only loads one sample or a small batch at each iteration
            # batch_idx = ind_rnd((1:pars.batchSize)+pars.batchSize*(batch-1));
            lower_index = pars['batchSize'] * batch
            upper_index = lower_index + pars['batchSize']
            batch_idx = ind_rnd[lower_index:upper_index]
            yt = trainingdata[:, batch_idx]
            ht = HMat[:, batch_idx]
            qt = QMat[:, batch_idx]
            # TODO: Review if in these cases it's mandatory to assign copies
            D = model['D']
            W = model['W']
            A = model['A']
            # sparse coding
            # S = L1QP_FeatureSign_Set(yt, D, gamma, lambda);

            S = lasso(
                yt,
                D if np.isfortran(D) else np.asfortranarray(D),
                **param
            )

            # compute the gradient of crf parameters
            grad_W = (1-mu)*(W@S - ht)@S.T + nu2*W
            grad_A = mu*(A@S - qt)@S.T + nu1*A
            grad_S1 = W.T @ (W@S - ht)  # gradient w.r.t S for 0.5*||H-WS||_2^2
            grad_S2 = A.T @ (A@S - qt)  # gradient w.r.t S for 0.5*||Q-AS||_2^2

            # compute the gradient of dictionary
            # find the active set and compute beta
            B1 = np.zeros((num_bases, pars['batchSize']), dtype=np.int)
            B2 = np.zeros((num_bases, pars['batchSize']), dtype=np.int)
            DtD = D.T@D
            for j in range(pars['batchSize']):
                active_set = np.array(np.nonzero(S[:, j] != 0)).ravel()
                # DtD = D(:,active_set)'*D(:,active_set) + gamma*eye(length(active_set));
                DtD_hat = DtD[active_set, active_set] + gamma*np.eye(active_set.shape[0])

                # DtD_inv = DtD\eye(length(active_set));
                DtD_inv = np.linalg.solve(DtD_hat, np.eye(active_set.shape[0]))

                B1[active_set, j] = (DtD_inv @ grad_S1[active_set, j]).T

                B2[active_set, j] = (DtD_inv @ grad_S2[active_set, j]).T

            grad_D = (1-mu)*(-D@B1@S.T + (yt - D@S)@B1.T) + mu * \
                (-D@B2@S.T + (yt - D@S)@B2.T)  # dD = -D*B*S' + (X - D*S)*B';

            # use yang's method
            # gfullMat = zeros([size(D),size(D,2)]);
            # [gMat, IDX] = sparseDerivative(D, full(S), yt);
            # gfullMat(:,IDX,IDX) = gMat;
            # gradSmat = repmat(reshape(grad_S1,[1 1 length(grad_S1)]),size(D));
            # grad_D = sum(gfullMat.*gradSmat,3);

            # update the learning rate
            rho_i = min(rho, rho*n0/(batch+1))

            # update model parameters
            D = D - rho_i*grad_D
            D = D / np.tile(np.linalg.norm(D, axis=0), (D.shape[0], 1))
            model['D'] = D

            W = W - rho_i*grad_W
            model['W'] = W

            A = A - rho_i*grad_A
            model['A'] = A

        # get statistics
        S = lasso(
            trainingdata,
            D if np.isfortran(D) else np.asfortranarray(D),
            **param
        )
        fobj = getobjective_lc(D, S, trainingdata, W, HMat, A, QMat, lambda_, mu)[0]
        # *** numpy.linalg.LinAlgError: Last 2 dimensions of the array must be square
        # stat['fobj_avg'][iter_] = fobj + 0.5*nu1*np.sum(W**2) + 0.5*nu2*np.sum(A**2)
        fobj_avg[iter_] = fobj + 0.5*nu1*np.sum(np.multiply(W, W)) + 0.5*nu2*np.sum(A**2)
        # filename = 'model_{}_{}_{}.npy'.format(iter_, num_bases, pars['dataset'])
        # full_path = os.path.join(generated_data_directory, filename)
        # ValueError: Object arrays cannot be saved when allow_pickle=False
        # the model is being saved in three different files to avoid
        # setting allow_pickle=True when trying to save the whole model
        # np.save(full_path, model, allow_pickle=True, fix_imports=False)
        for key, value in model.items():
            filename = '{}_{}_{}_{}.npy'.format(key, iter_, num_bases, pars['dataset'])
            full_path = os.path.join(generated_data_directory, filename)
            np.save(full_path, value, allow_pickle=False, fix_imports=False)

        toc = time.time()
        print('Iter = {}, Elapsed Time = {}\n'.format(iter_, toc-tic))

    stat_filename = 'stat_{}_{}.json'.format(num_bases, pars['dataset'])
    stat_full_path = os.path.join(generated_data_directory, stat_filename)
    # saving as JSON to avoid using pickle
    with open(stat_full_path, 'w') as file_:
        json.dump(fobj_avg, file_)

    return model, fobj_avg


def load_model(iteration, num_bases, dataset):
    """
    Loads the matrices for the especified iteration, num_bases and dataset and returns it.

    Args:
        iteration  (int): iteration when the matrix was saved
        numb_bases (int): num_bases used
        dataset    (str): dataset name

    Returns:
        dict(D=np.ndarray, W=np.ndarray, A=np.dnarray)
    """
    assert isinstance(iteration, int)
    assert isinstance(num_bases, int)
    assert isinstance(dataset, str)

    model = dict()
    matrices = [
        'D',  # dictionary
        'W',  # classifier
        'A',  # transform matrix
    ]

    for matrix in matrices:
        filename = '{}_{}_{}_{}.npy'.format(matrix, iteration, num_bases, dataset)
        full_path = os.path.join(generated_data_directory, filename)

        if os.path.isfile(full_path):
            model[matrix] = np.load(full_path, allow_pickle=False, fix_imports=False)

    return model


def load_stats(num_bases, dataset):
    """
    Loads the stats generated during training with num_bases and dataset
    Args:
        iteration  (int): iteration when the matrix was saved
        numb_bases (int): num_bases used

    Returns
        dict(iter0=fobj_avg, iter1=fobj_avg, ...)
    """
    assert isinstance(num_bases, int)
    assert isinstance(dataset, str)

    stat_filename = 'stat_{}_{}.json'.format(num_bases, dataset)
    stat_full_path = os.path.join(generated_data_directory, stat_filename)

    with open(stat_full_path, 'r') as file_:
        fobj_avg = json.load(file_)

    fobj_avg = {int(k): v for k, v in fobj_avg.items()}

    return fobj_avg


# uncomment
# tic = time.time()
# model, fobj_avg = onlineDictionaryLearning(pars, trainingsubset, H_train_subset, Q_train)
# toc = time.time()
# print('done! it took {} seconds'.format(toc-tic))
# # done! it took 1577.4330954551697 seconds


def classification(D, W, data, Hlabel, sparsity):
    """
    Performs the classification

    Args:
        D          (np.ndarray): learned dictionary
        W          (np.ndarray): learned classifier parameters
        data       (np.ndarray): testing features
        Hlabel     (np.ndarray): labels matrix for testing feature
        iterations        (int): iterations for KSVD
        sparsity          (int): sparsity threshold

    Returns:
        prediction (np.ndarray): predicted labels for testing features
        accuracy        (float): classification accuracy
        err        (np.ndarray): misclassfication information
                                 [errid, featureid, groundtruth-label, predicted-label]

    Author: Zhuolin Jiang (zhuolin@umiacs.umd.edu)
    Date: 10-16-2011
    """
    assert isinstance(D, np.ndarray)
    assert isinstance(W, np.ndarray)
    assert isinstance(data, np.ndarray)
    assert isinstance(Hlabel, np.ndarray)
    assert isinstance(iterations, int)
    assert isinstance(sparsity, int)

    # sparse coding
    # http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams005.html#sec12
    Gamma = omp(data, D if np.isfortran(D) else np.asfortranarray(D), sparsity)

    # classify process
    errnum = 0
    err = np.empty((0, 4))
    prediction = np.empty((1, 0), dtype=np.int)

    for featureid in range(data.shape[1]):
        spcode = Gamma[:, featureid]
        score_est = W @ spcode
        score_gt = Hlabel[:, featureid]
        maxind_est = score_est.argmax()  # classifying
        maxind_gt = score_gt.argmax()
        prediction = np.c_[prediction, maxind_est]

        if maxind_est != maxind_gt:
            errnum += 1
            err = np.r_[err, [[errnum, featureid, maxind_gt, maxind_est]]]

    accuracy = (data.shape[1] - errnum)/data.shape[1]

    return prediction, accuracy, err


accuracy_list = []
fobj_avg = load_stats(pars['numBases'], pars['dataset'])

for ii in range(pars['maxIters']):
    model = load_model(ii, pars['numBases'], pars['dataset'])
    D1 = model['D']
    W1 = model['W']

    # classification
    accuracy_list.append(classification(D1, W1, testingsubset, H_test_subset, sparsitythres)[1])
    print('\nFinal recognition rate for OnlineDL is : {} , objective function value: {}'.format(
        accuracy_list[ii], fobj_avg[ii]))

    if not bool(ii % 10):
        print('\n')

accuracy_list = np.asarray(accuracy_list)

print('Best recognition rate for OnlineDL is {} at iteration {}\n'.format(
    accuracy_list.max(), accuracy_list.argmax()))


# plot the objective function values for all iterations
# plt.clf()
# plt.plot(list(fobj_avg.keys()), list(fobj_avg.values()), 'mo--', linewidth=2)
# plt.xlabel('Iterations')
# plt.ylabel('Average objective function value')
# plt.xticks(list(range(0, 20)), list(range(1, 21)))
# plt.show()

# plt.clf()
# plt.plot(accuracy_list, 'rs--', linewidth=2)
# plt.xticks(list(range(0, 20)), list(range(1, 21)))
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.show()
# Best recognition rate for OnlineDL is 0.710552268244576 at iteration 3
# Best recognition rate for OnlineDL is 0.7140039447731755 at iteration 0
# the paper report for LC-KSVD1 and LC-KSVD2 73.4 and 73.6 respectively for 30 samples
# per class
# all the evaluations were done using spatial pyramid features
# They used 510, 1,020, 1,530, 2,040, 2,550, and 3,060
# dictionary items/sizes, respectively, for 5, 10, 15, 20, 25, and
# 30 training samples per category.
