# -*- coding: utf-8 -*-
""" utils/datasets/spatialpyramidfeatures4caltech101 """

import numpy as np
from scipy.io import loadmat

import settings


class DBhandler:
    """
    Handler for SpatialPyramidFeatures4Caltech101 dataset

    Usage:
        train_feats, train_labels, test_feats, test_labels = DBhandler()()
    """

    def __init__(self):
        """ Loads the data """
        self.data = loadmat(settings.FULL_DATASET_PATH)
        # data['filenameMat'].shape (1, 102)
        # data['featureMat'].shape (3000, 9144)
        # data['labelMat'].shape (102, 9144)

    def __call__(self):
        """ functor call """
        return self.__get_training_testing_subsets()

    @staticmethod
    def __get_training_testing_sets(feat_matrix, label_matrix, num_per_class):
        """
        Obtain training and testing features by random sampling

        Args:
            feat_matrix   (np.ndarray): input features
            label_matrix  (np.ndarray): label matrix for input features
            num_per_class        (int): number of training samples from each category

        Return:
            train_feats  (np.ndarray): training features
            train_labels (np.ndarray): label matrix for training features
            test_feats   (np.ndarray): testing features
            test_labels  (np.ndarray): label matrix for testing features
        """
        assert isinstance(feat_matrix, np.ndarray)
        assert isinstance(label_matrix, np.ndarray)
        assert isinstance(num_per_class, int)

        num_class = label_matrix.shape[0]  # number of objects
        test_feats = np.empty((feat_matrix.shape[0], 0))
        test_labels = np.empty((label_matrix.shape[0], 0))
        train_feats = np.empty((feat_matrix.shape[0], 0))
        train_labels = np.empty((label_matrix.shape[0], 0))

        for classid in range(num_class):
            col_ids = np.array(np.nonzero(label_matrix[classid, :] == 1)).ravel()
            data_ids = np.array(np.nonzero(np.sum(feat_matrix[:, col_ids]**2, axis=0) > 1e-6))\
                         .ravel()
            trainids = col_ids[np.random.choice(data_ids, num_per_class, replace=False)]
            testids = np.setdiff1d(col_ids, trainids)
            test_feats = np.c_[test_feats, feat_matrix[:, testids]]
            test_labels = np.c_[test_labels, label_matrix[:, testids]]
            train_feats = np.c_[train_feats, feat_matrix[:, trainids]]
            train_labels = np.c_[train_labels, label_matrix[:, trainids]]

        return train_feats, train_labels, test_feats, test_labels

    def __get_training_testing_subsets(self):
        """ Returns the training and testing subsets  """
        # getting training and testing data
        train_feats, train_labels, test_feats, test_labels = self.__get_training_testing_sets(
            self.data['featureMat'], self.data['labelMat'], settings.PARS['ntrainsamples'])
        # test_feats (3000, 6084)
        # test_labels (102, 6084)
        # train_feats (3000, 3060)
        # train_labels (102, 3060)

        # getting the subsets of training data and testing data
        # it is related to the variable 'personnumber'
        labelvector_train, _ = train_labels.nonzero()  # 3060
        labelvector_test, _ = test_labels.nonzero()  # 6084
        trainsampleid = np.nonzero(labelvector_train <= settings.PERSON_NUMBER)[0]  # 3060
        testsampleid = np.nonzero(labelvector_test <= settings.PERSON_NUMBER)[0]  # 6084
        train_subset_feats = train_feats[:, trainsampleid]  # 3000, 3060
        test_subset_feats = test_feats[:, testsampleid]  # 3000, 6084
        train_subset_labels = train_labels[: settings.PERSON_NUMBER+1, trainsampleid]  # 102, 3060
        test_subset_labels = test_labels[: settings.PERSON_NUMBER+1, testsampleid]  # 102, 6084

        return train_subset_feats, train_subset_labels, test_subset_feats, test_subset_labels
