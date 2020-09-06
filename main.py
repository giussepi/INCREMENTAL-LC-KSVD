# -*- coding: utf-8 -*-
""" main """

import json
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from spams import trainDL, lasso, omp

import settings
from utils.datasets.spatialpyramidfeatures4caltech101 import DBhandler


###############################################################################
#                     Verified with matlabscript till here                    #
###############################################################################

##########
# Incremental Dictionary learning

trainingsubset, H_train_subset, testingsubset, H_test_subset = DBhandler()()

# np.save(os.path.join(settings.GENERATED_DATA_DIRECTORY, 'trainingsubset.npy'),
#         trainingsubset, allow_pickle=False, fix_imports=False)
# np.save(os.path.join(settings.GENERATED_DATA_DIRECTORY, 'H_train_subset.npy'),
#         H_train_subset, allow_pickle=False, fix_imports=False)
# np.save(os.path.join(settings.GENERATED_DATA_DIRECTORY, 'testingsubset.npy'),
#         testingsubset, allow_pickle=False, fix_imports=False)
# np.save(os.path.join(settings.GENERATED_DATA_DIRECTORY, 'H_test_subset.npy'),
#         H_test_subset, allow_pickle=False, fix_imports=False)


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

        # Initilization for LC-KSVD (perform KSVD in each class)
        Dpart = training_feats[:, col_ids[np.random.choice(data_ids, numPerClass, replace=False)]]
        param1['D'] = Dpart  # initial dictionary
        Dpart = trainDL(training_feats[:, col_ids[data_ids]], **param1)
        Dinit = np.c_[Dinit, Dpart]
        labelvector = np.zeros((numClass, 1))  # eyes
        labelvector[classid] = 1
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
# [Dinit, Winit, Tinit, Q_train] = paramterinitialization(trainingsubset, H_train_subset, settings.PARS)


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


def onlineDictionaryLearning(dinit, tinit, winit, pars, trainingdata, HMat, QMat):
    """
    online dictionary learning using stochastic gradient descent algorithm
    Args:
        dinit (np.ndarray): initial dictionary
        tinit (np.ndarray): initial transform matrix
        winit (np.ndarray): initial classifier
        pars        (dict): learning paramters
           'mu': reguliarization parameter
           'nu1': reguliarization parameter
           'nu2': reguliarization parameter
           'maxIters': iteration number
           'rho': learning rate parameter
        trainingdata (np.ndarray): input training des. with size of n X N
        HMat         (np.ndarray): label matrix of training des. with size of m X N
        QMat         (np.ndarray): optimal code matrix of training des. with size of K X N

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
        D=dinit,  # dictionary
        A=tinit,  # transform matrix
        W=winit,  # classifier
    )

    param = {
        'lambda1': pars['lambda_'],
        'lambda2': 0,
        'mode': 2
    }

    # crf iterations
    fobj_avg = dict()

    if not os.path.isdir(settings.GENERATED_DATA_DIRECTORY):
        os.mkdir(settings.GENERATED_DATA_DIRECTORY)

    for iter_ in range(num_iters):
        tic = time.time()
        # Take a random permutation of the samples
        filename = 'permute_{}_{}_{}.npy'.format(iter_, num_bases, pars['dataset'])
        full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, filename)

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
            full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, filename)
            np.save(full_path, value, allow_pickle=False, fix_imports=False)

        toc = time.time()
        print('Iter = {}, Elapsed Time = {}\n'.format(iter_, toc-tic))

    stat_filename = 'stat_{}_{}.json'.format(num_bases, pars['dataset'])
    stat_full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, stat_filename)
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
        'A',  # transform matrix
        'W',  # classifier
    ]

    for matrix in matrices:
        filename = '{}_{}_{}_{}.npy'.format(matrix, iteration, num_bases, dataset)
        full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, filename)

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
    stat_full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, stat_filename)

    with open(stat_full_path, 'r') as file_:
        fobj_avg = json.load(file_)

    fobj_avg = {int(k): v for k, v in fobj_avg.items()}

    return fobj_avg


# uncomment
# tic = time.time()
# model, fobj_avg = onlineDictionaryLearning(Dinit, Tinit, Winit, settings.PARS, trainingsubset, H_train_subset, Q_train)
# toc = time.time()
# print('done! it took {} seconds'.format(toc-tic))
# done! it took 1577.4330954551697 seconds


def classification(D, W, data, Hlabel, sparsity):
    """
    Performs the classification

    Args:
        D          (np.ndarray): learned dictionary
        W          (np.ndarray): learned classifier parameters
        data       (np.ndarray): testing features
        Hlabel     (np.ndarray): labels matrix for testing feature
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
fobj_avg = load_stats(settings.PARS['numBases'], settings.PARS['dataset'])

for ii in range(settings.PARS['maxIters']):
    model = load_model(ii, settings.PARS['numBases'], settings.PARS['dataset'])
    D1 = model['D']
    W1 = model['W']

    # classification
    accuracy_list.append(classification(D1, W1, testingsubset, H_test_subset, settings.SPARSITYTHRES)[1])
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
# Best recognition rate for OnlineDL is 0.7218934911242604 at iteration 1
# Best recognition rate for OnlineDL is 0.7345496383957922 at iteration 1
# Best recognition rate for OnlineDL is 0.7355358316896778 at iteration 1
# the paper report for LC-KSVD1 and LC-KSVD2 73.4 and 73.6 respectively for 30 samples
# per class
# all the evaluations were done using spatial pyramid features
# They used 510, 1,020, 1,530, 2,040, 2,550, and 3,060
# dictionary items/sizes, respectively, for 5, 10, 15, 20, 25, and
# 30 training samples per category.
