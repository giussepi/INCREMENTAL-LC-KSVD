# -*- coding: utf-8 -*-
""" models/ilc_ksvd """

import json
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from spams import trainDL, lasso, omp

import settings
from utils.datasets.spatialpyramidfeatures4caltech101 import DBhandler


class ILCksvd:
    """
    Implements incremental LC-LSVD

    Usage:
        ilc_ksvd = ILCksvd(DBhandler)
        ilc_ksvd.train()
        ilc_ksvd.test(plot=True)
    """

    def __init__(self, db_handler_class=DBhandler):
        """
        Gets the training and testing datasets

        Args:
            db_handler_class (type): Dataset handler class

        """
        assert isinstance(db_handler_class, type)

        self.train_feats, self.train_labels, self.test_feats, self.test_labels = \
            db_handler_class()()

    def parameter_initialization(self):
        """
        parameter initialization for incremental dictionary learning

        Returns:
            Dinit (np.ndarray): initial dictionary
            Winit (np.ndarray): initial classifier
            Tinit (np.ndarray): initial transform matrix
            Q     (np.ndarray): optimal code matrix of training des. with size of K X N
        """
        dictsize = settings.PARS.get('numBases')
        numClass = self.train_labels.shape[0]  # number of objects
        Dinit = np.empty((self.train_feats.shape[0], 0))  # for C-Ksvd and D-Ksvd
        dictLabel = np.empty((numClass, 0), dtype=np.int)
        numPerClass = dictsize//numClass
        param1 = {
            'mode': 2,
            'K': settings.PARS.get('numBases'),  # size of the dictionary
            'lambda1': settings.PARS.get('lambda_'),
            'lambda2': 0,
            'iter': settings.PARS.get('iterationini')
        }
        param2 = {
            'lambda1': settings.PARS.get('lambda_'),
            'lambda2': 0,
            'mode': 2
        }

        # for classid in range(numClass):
        #     labelvector = np.zeros((numClass, 1), dtype=np.int)
        #     labelvector[classid] = 1
        #     dictLabel = np.c_[dictLabel, np.tile(labelvector, (1, numPerClass))]

        for classid in range(numClass):
            col_ids = np.array(np.nonzero(self.train_labels[classid, :] == 1)).ravel()
            # ensure no zero data elements are chosen
            data_ids = np.array(np.nonzero(np.sum(self.train_feats[:, col_ids]**2, axis=0) > 1e-6)).ravel()

            # Initilization for LC-KSVD (perform KSVD in each class)
            Dpart = self.train_feats[:, col_ids[np.random.choice(data_ids, numPerClass, replace=False)]]
            param1['D'] = Dpart  # initial dictionary
            Dpart = trainDL(self.train_feats[:, col_ids[data_ids]], **param1)
            Dinit = np.c_[Dinit, Dpart]
            labelvector = np.zeros((numClass, 1), dtype=np.int)
            labelvector[classid] = 1
            dictLabel = np.c_[dictLabel, np.tile(labelvector, (1, numPerClass))]

        param1['D'] = np.asfortranarray(Dinit)  # initial dictionary
        # RuntimeError: matrix arg 10 must be a 2d double Fortran Array
        Dinit = trainDL(self.train_feats, **param1)

        Xinit = lasso(self.train_feats, Dinit, **param2)

        # learning linear classifier parameters
        tmp = np.linalg.inv(Xinit@Xinit.T+np.eye(*(Xinit@Xinit.T).shape))@Xinit
        Winit = tmp@self.train_labels.T
        Winit = Winit.T

        Q = np.zeros((dictsize, self.train_feats.shape[1]))  # energy matrix

        for frameid in range(self.train_feats.shape[1]):
            label_training = self.train_labels[:, frameid]
            maxid1 = label_training.argmax(0)

            for itemid in range(Dinit.shape[1]):
                label_item = dictLabel[:, itemid]
                maxid2 = label_item.argmax(0)

                if maxid1 == maxid2:
                    Q[itemid, frameid] = 1

        Tinit = tmp@Q.T
        Tinit = Tinit.T

        return Dinit, Winit, Tinit, Q

    @staticmethod
    def get_objective_lc(D, S, Y, W, H, A, Q, lambda_, mu):
        """  """
        E1 = D.astype(np.float64)@S - Y.astype(np.float64)
        E2 = A.astype(np.float64)@S - Q.astype(np.float64)
        E3 = W.astype(np.float64)@S - H.astype(np.float64)

        fresidue1 = np.sum(E1**2)  # reconstruction error
        fresidue2 = np.sum(E2**2)  # optimal sparse code error
        # TODO: Review why this happens, I believe it's an issue related with W.
        # maybe it's calculated in a lazy way I'm not sure.....
        # numpy.linalg.LinAlgError: Last 2 dimensions of the array must be square
        # fresidue3 = np.sum(E3**2)  # classification error
        fresidue3 = np.sum(np.multiply(E3, E3))
        freg = lambda_*np.sum(np.absolute(S))
        fobj = 0.5*(mu*fresidue2 + (1-mu)*fresidue3)

        return fobj, fresidue1, fresidue2, fresidue3, freg

    def online_dictionary_learning(self, dinit, tinit, winit, pars, trainingdata, HMat, QMat):
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
            model (dict): learned paramters
                'D' (np.ndarray): learned dictionary
                'A' (np.ndarray): learned transform matrix
                'W' (np.ndarray): learned classifier
            fobj_avg (dict): average objective function value
        """
        num_images = trainingdata.shape[1]
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

        for iter_ in range(pars['maxIters']):
            tic = time.time()
            # Take a random permutation of the samples
            filename = 'permute_{}_{}_{}.npy'.format(iter_, pars['numBases'], pars['dataset'])
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
                grad_W = (1-pars['mu'])*(W@S - ht)@S.T + pars['nu2']*W
                grad_A = pars['mu']*(A@S - qt)@S.T + pars['nu1']*A
                grad_S1 = W.T @ (W@S - ht)  # gradient w.r.t S for 0.5*||H-WS||_2^2
                grad_S2 = A.T @ (A@S - qt)  # gradient w.r.t S for 0.5*||Q-AS||_2^2

                # compute the gradient of dictionary
                # find the active set and compute beta
                B1 = np.zeros((pars['numBases'], pars['batchSize']), dtype=np.int)
                B2 = np.zeros((pars['numBases'], pars['batchSize']), dtype=np.int)
                DtD = D.T@D
                for j in range(pars['batchSize']):
                    active_set = np.array(np.nonzero(S[:, j] != 0)).ravel()
                    # DtD = D(:,active_set)'*D(:,active_set) + gamma*eye(length(active_set));
                    DtD_hat = DtD[active_set, active_set] + pars['gamma']*np.eye(active_set.shape[0])

                    # DtD_inv = DtD\eye(length(active_set));
                    DtD_inv = np.linalg.solve(DtD_hat, np.eye(active_set.shape[0]))

                    B1[active_set, j] = (DtD_inv @ grad_S1[active_set, j]).T

                    B2[active_set, j] = (DtD_inv @ grad_S2[active_set, j]).T

                grad_D = (1-pars['mu'])*(-D@B1@S.T + (yt - D@S)@B1.T) + pars['mu'] * \
                    (-D@B2@S.T + (yt - D@S)@B2.T)  # dD = -D*B*S' + (X - D*S)*B';

                # use yang's method
                # gfullMat = zeros([size(D),size(D,2)]);
                # [gMat, IDX] = sparseDerivative(D, full(S), yt);
                # gfullMat(:,IDX,IDX) = gMat;
                # gradSmat = repmat(reshape(grad_S1,[1 1 length(grad_S1)]),size(D));
                # grad_D = sum(gfullMat.*gradSmat,3);

                # update the learning rate
                rho_i = min(pars['rho'], pars['rho']*n0/(batch+1))

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
            fobj = self.get_objective_lc(D, S, trainingdata, W, HMat, A, QMat, pars['lambda_'], pars['mu'])[0]
            # *** numpy.linalg.LinAlgError: Last 2 dimensions of the array must be square
            # stat['fobj_avg'][iter_] = fobj + 0.5*nu1*np.sum(W**2) + 0.5*nu2*np.sum(A**2)
            fobj_avg[iter_] = fobj + 0.5*pars['nu1']*np.sum(np.multiply(W, W)) + 0.5*pars['nu2']*np.sum(A**2)
            # filename = 'model_{}_{}_{}.npy'.format(iter_, num_bases, pars['dataset'])
            # full_path = os.path.join(generated_data_directory, filename)
            # ValueError: Object arrays cannot be saved when allow_pickle=False
            # the model is being saved in three different files to avoid
            # setting allow_pickle=True when trying to save the whole model
            # np.save(full_path, model, allow_pickle=True, fix_imports=False)
            for key, value in model.items():
                filename = '{}_{}_{}_{}.npy'.format(key, iter_, pars['numBases'], pars['dataset'])
                full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, filename)
                np.save(full_path, value, allow_pickle=False, fix_imports=False)

            toc = time.time()
            print('Iter = {}, Elapsed Time = {}'.format(iter_, toc-tic))

        stat_filename = 'stat_{}_{}.json'.format(pars['numBases'], pars['dataset'])
        stat_full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, stat_filename)
        # saving as JSON to avoid using pickle
        with open(stat_full_path, 'w') as file_:
            json.dump(fobj_avg, file_)

        return model, fobj_avg

    def train(self):
        """ Gets the initialization parameters and trains the model """
        print("Paremeter initialization")
        Dinit, Winit, Tinit, Q_train = self.parameter_initialization()
        print("Completed")
        print('Incremental dictionary learning...')
        tic = time.time()
        _, _ = self.online_dictionary_learning(Dinit, Tinit, Winit, settings.PARS,
                                               self.train_feats, self.train_labels, Q_train)
        toc = time.time()
        print('nIncremental dictionary learning completed! it took {} seconds'.format(toc-tic))

    def classification(self, D, W):
        """
        Performs the classification

        Args:
            D          (np.ndarray): learned dictionary
            W          (np.ndarray): learned classifier parameters

        Returns:
            prediction (np.ndarray): predicted labels for testing features
            accuracy        (float): classification accuracy
            err        (np.ndarray): misclassfication information
                                     [errid, featureid, groundtruth-label, predicted-label]
        """
        assert isinstance(D, np.ndarray)
        assert isinstance(W, np.ndarray)
        assert isinstance(self.test_feats, np.ndarray)
        assert isinstance(self.test_labels, np.ndarray)
        assert isinstance(settings.SPARSITYTHRES, int)

        # sparse coding
        # http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams005.html#sec12
        Gamma = omp(
            self.test_feats,
            D if np.isfortran(D) else np.asfortranarray(D), settings.SPARSITYTHRES
        )

        # classify process
        errnum = 0
        err = np.empty((0, 4))
        prediction = np.empty((1, 0), dtype=np.int)

        for featureid in range(self.test_feats.shape[1]):
            spcode = Gamma[:, featureid]
            score_est = W @ spcode
            score_gt = self.test_labels[:, featureid]
            maxind_est = score_est.argmax()  # classifying
            maxind_gt = score_gt.argmax()
            prediction = np.c_[prediction, maxind_est]

            if maxind_est != maxind_gt:
                errnum += 1
                err = np.r_[err, [[errnum, featureid, maxind_gt, maxind_est]]]

        accuracy = (self.test_feats.shape[1] - errnum)/self.test_feats.shape[1]

        return prediction, accuracy, err

    @staticmethod
    def load_model(iteration):
        """
        Loads the matrices for the especified iteration, num_bases and dataset and returns it.

        Args:
            iteration  (int): iteration when the matrix was saved

        Returns:
            dict(D=np.ndarray, W=np.ndarray, A=np.dnarray)
        """
        assert isinstance(iteration, int)
        assert isinstance(settings.PARS['numBases'], int)
        assert isinstance(settings.PARS['dataset'], str)

        model = dict()
        matrices = [
            'D',  # dictionary
            'A',  # transform matrix
            'W',  # classifier
        ]

        for matrix in matrices:
            filename = '{}_{}_{}_{}.npy'.format(
                matrix, iteration, settings.PARS['numBases'], settings.PARS['dataset'])
            full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, filename)

            if os.path.isfile(full_path):
                model[matrix] = np.load(full_path, allow_pickle=False, fix_imports=False)

        return model

    @staticmethod
    def load_stats():
        """
        Loads the stats generated during training with num_bases and dataset

        Returns
            dict(iter0=fobj_avg, iter1=fobj_avg, ...)
        """
        assert isinstance(settings.PARS['numBases'], int)
        assert isinstance(settings.PARS['dataset'], str)

        stat_filename = 'stat_{}_{}.json'.format(
            settings.PARS['numBases'], settings.PARS['dataset'])
        stat_full_path = os.path.join(settings.GENERATED_DATA_DIRECTORY, stat_filename)

        with open(stat_full_path, 'r') as file_:
            fobj_avg = json.load(file_)

        fobj_avg = {int(k): v for k, v in fobj_avg.items()}

        return fobj_avg

    def test(self, plot=False):
        """
        * Test the classifier and prints the results]
        * If plot=True the results are plotted
        """
        accuracy_list = []
        fobj_avg = self.load_stats()

        for ii in range(settings.PARS['maxIters']):
            model = self.load_model(ii)
            D1 = model['D']
            W1 = model['W']

            # classification
            accuracy_list.append(self.classification(D1, W1)[1])
            print(
                'Final recognition rate for OnlineDL is : {} , objective function value: {}'
                .format(accuracy_list[ii], fobj_avg[ii])
            )

        accuracy_list = np.asarray(accuracy_list)

        print('Best recognition rate for OnlineDL is {} at iteration {}'.format(
            accuracy_list.max(), accuracy_list.argmax()))

        if plot:
            # plot the objective function values for all iterations
            plt.clf()
            plt.plot(list(fobj_avg.keys()), list(fobj_avg.values()), 'mo--', linewidth=2)
            plt.xlabel('Iterations')
            plt.ylabel('Average objective function value')
            plt.xticks(list(range(0, 20)), list(range(1, 21)))
            plt.show()

            plt.clf()
            plt.plot(accuracy_list, 'rs--', linewidth=2)
            plt.xticks(list(range(0, 20)), list(range(1, 21)))
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.show()
