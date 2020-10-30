# -*- coding: utf-8 -*-
""" utils/datasets/generic """

import json

import numpy as np

import settings
from utils.datasets.exceptions import DatasetZeroElementFound
from utils.utils import Normalizer


class DBhandler:
    """
    Generic dataset handler

    Usage:
        train_feats, train_labels, test_feats, test_labels = DBhandler(Normalizer.NONE)()
    """

    def __init__(self, normalizer=Normalizer.NONE):
        """
        Loads the training and testing datasets

        Args:
            normalizer (Normalizer option): Normalization to apply
        """
        with open(settings.TRAINING_DATA_DIRECTORY_DATASET_PATH, 'r') as file_:
            self.training_data = json.load(file_)
            self.to_numpy(self.training_data)

            if normalizer in Normalizer.CHOICES[:4]:
                # sample normalization
                self.training_data['codes'] = Normalizer()(
                    normalizer, data=self.training_data['codes'].T).T
            else:
                # feature scaling
                scaler, self.training_data['codes'] = Normalizer()(
                    normalizer, data=self.training_data['codes'].T)
                self.training_data['codes'] = self.training_data['codes'].T

            # sorting dataset
            self.training_data['codes'], self.training_data['labels'] = self.sort_dataset(
                self.training_data['codes'], self.training_data['labels'])
            self.training_data['labels'] = self.training_data['labels'].astype(np.float64)

        with open(settings.TESTING_DATA_DIRECTORY_DATASET_PATH, 'r') as file_:
            self.testing_data = json.load(file_)
            self.to_numpy(self.testing_data)

            if normalizer in Normalizer.CHOICES[:4]:
                # sample normalization
                self.testing_data['codes'] = Normalizer()(
                    normalizer, data=self.testing_data['codes'].T).T
            else:
                # feature scaling
                self.testing_data['codes'] = Normalizer()(
                    normalizer, data=self.testing_data['codes'].T, fitted_scaler=scaler)[1].T

            # sorting dataset
            self.testing_data['codes'], self.testing_data['labels'] = self.sort_dataset(
                self.testing_data['codes'], self.testing_data['labels'])
            self.testing_data['labels'] = self.testing_data['labels'].astype(np.float64)

    def __call__(self):
        """ functor call """
        return self.__get_training_testing_sets()

    @staticmethod
    def sort_dataset(feats, labels):
        """
        Sorts the features and labels in ascending order and returns them

        Args:
            feats  (np.ndarray): feature matrix with shape (num features, num samples)
            labels (np.ndarray): label matix with shape (num labels, num samples)

        Returns:
            sorted feats (np.ndarray), sorted labels (np.ndarray)
        """
        assert isinstance(feats, np.ndarray)
        assert isinstance(labels, np.ndarray)

        num_classes = labels.shape[0]
        sorted_feats = np.empty((feats.shape[0], 0))
        sorted_labels = np.empty((labels.shape[0], 0))
        rng = np.random.default_rng()

        for class_id in range(num_classes):
            col_ids = np.array(np.nonzero(labels[class_id, :] == 1)).ravel()
            # making sure there's zero elements
            data_ids = np.array(np.nonzero(np.sum(feats[:, col_ids]**2, axis=0) > 1e-6)).ravel()

            # Raising an error if any zero lement is found
            if col_ids.shape[0] != data_ids.shape[0]:
                raise DatasetZeroElementFound

            rng.shuffle(data_ids, axis=0)
            ids = col_ids[data_ids]
            sorted_feats = np.c_[sorted_feats, feats[:, ids]]
            sorted_labels = np.c_[sorted_labels, labels[:, ids]]

        return sorted_feats, sorted_labels

    @staticmethod
    def to_numpy(data):
        """  """
        for key in data:
            data[key] = np.asfortranarray(data[key])

    def __get_training_testing_sets(self):
        """ Returns training and testing data """
        return self.training_data['codes'], self.training_data['labels'], \
            self.testing_data['codes'], self.testing_data['labels']
