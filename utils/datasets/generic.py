# -*- coding: utf-8 -*-
""" utils/datasets/generic """

import json

import numpy as np

import settings


class DBhandler:
    """
    Generic dataset handler

    Usage:
        train_feats, train_labels, test_feats, test_labels = DBhandler()()
    """

    def __init__(self):
        """ Loads the training and testing datasets """
        with open(settings.TRAINING_DATA_DIRECTORY_DATASET_PATH, 'r') as file_:
            self.training_data = json.load(file_)
            self.to_numpy(self.training_data)

        with open(settings.TESTING_DATA_DIRECTORY_DATASET_PATH, 'r') as file_:
            self.testing_data = json.load(file_)
            self.to_numpy(self.testing_data)

    def __call__(self):
        """ functor call """
        return self.__get_training_testing_sets()

    @staticmethod
    def to_numpy(data):
        """  """
        for key in data:
            data[key] = np.array(data[key])

    def __get_training_testing_sets(self):
        """ Returns training and testing data """
        return self.training_data['codes'], self.training_data['labels'], \
            self.testing_data['codes'], self.testing_data['labels']
