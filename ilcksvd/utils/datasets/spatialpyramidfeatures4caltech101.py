# -*- coding: utf-8 -*-
""" utils/datasets/spatialpyramidfeatures4caltech101 """

import numpy as np
from scipy.io import loadmat

import settings

from ilcksvd.utils.utils import Normalizer


class DBhandler:
    """
    Handler for SpatialPyramidFeatures4Caltech101 dataset

    Usage:
        train_feats, train_labels, test_feats, test_labels = DBhandler(Normalizer.NONE)()
    """

    def __init__(self, normalizer=Normalizer.NONE):
        """
        Loads spatial pyramids features from caltech101 from .mat file

        Args:
            normalizer (Normalizer option): Normalization to apply
        """
        self.normalizer = normalizer
        self.data = loadmat(settings.FULL_DATASET_PATH)
        # data['filenameMat'].shape (1, 102)
        # data['featureMat'].shape (3000, 9144)
        # data['labelMat'].shape (102, 9144)

    def __call__(self):
        """ functor call """
        return self.__get_training_testing_subsets()

    def __get_training_testing_sets(self, feat_matrix, label_matrix, num_per_class):
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

        if self.normalizer in Normalizer.CHOICES[:4]:
            # sample normalization
            train_feats = Normalizer()(self.normalizer, data=train_feats.T).T
            test_feats = Normalizer()(self.normalizer, data=test_feats.T).T
        else:
            # feature scaling
            scaler, train_feats = Normalizer()(self.normalizer, data=train_feats.T)
            train_feats = train_feats.T
            test_feats = Normalizer()(self.normalizer, data=test_feats.T, fitted_scaler=scaler)[1].T

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
        labelvector_train, _ = train_labels.nonzero()  # 3060
        labelvector_test, _ = test_labels.nonzero()  # 6084
        trainsampleid = np.nonzero(labelvector_train <= settings.CLASS_NUMBER)[0]  # 3060
        testsampleid = np.nonzero(labelvector_test <= settings.CLASS_NUMBER)[0]  # 6084
        train_subset_feats = train_feats[:, trainsampleid]  # 3000, 3060
        test_subset_feats = test_feats[:, testsampleid]  # 3000, 6084
        train_subset_labels = train_labels[: settings.CLASS_NUMBER, trainsampleid]  # 102, 3060
        test_subset_labels = test_labels[: settings.CLASS_NUMBER, testsampleid]  # 102, 6084

        return train_subset_feats, train_subset_labels, test_subset_feats, test_subset_labels
