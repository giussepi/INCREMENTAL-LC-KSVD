# -*- coding: utf-8 -*-
""" utils/utils """

import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler


def normcols(matrix):
    """
    Returns an array with columns normalised
    Args:
        matrix (np.ndarray): matrix
    Returns:
        np.ndarray
    """
    assert isinstance(matrix, np.ndarray)

    return matrix/np.linalg.norm(matrix, axis=0)


class Normalizer:
    """
    Holds several normalizing options

    Usage:
        # sample normalization
        train_data = Normalizer()(Normalizer.L2_NORM, data=train_data)
        test_data = Normalizer()(Normalizer.L2_NORM, data=test_data)

        # feature scaling
        scaler, train_data = Normalizer()(Normalizer.STANDARDIZE, data=train_data)
        test_data = Normalizer(Normalizer, data=test_data, fitted_scaler=scaler)[1]
    """
    NONE = 'none'
    MAX_NORM = 'max_norm'
    L1_NORM = 'l1_norm'
    L2_NORM = 'l2_norm'
    STANDARDIZE = 'standardize'
    NORMALIZE = 'normalize'

    CHOICES = (NONE, MAX_NORM, L1_NORM, L2_NORM, STANDARDIZE, NORMALIZE)

    def __call__(self, option, **kwargs):
        """
        Functor call

        Gets the method based on the provided option, calls it using the provided kwargs,
        then returns its results

        Args:
            option (Normalizer.CHOICES): one of the implemented options

        Kwargs:
            keyword arguments for the method called

        Returns:
            Results from called method (could be the normalized data or scaler, normalized_data)

        """
        return self.get_normalizer(option, **kwargs)

    @classmethod
    def none(cls, data):
        """
        Returns the same data

        Args:
            data (np.ndarray): matrix with shape (n_samples, n_features)

        Returns:
            np.ndarray
        """
        assert isinstance(data, np.ndarray)

        return data

    @classmethod
    def lp_norm(cls, data, option):
        """
        Performs unit vector normalization over the samples and returns the normalized data

        Args:
            data (np.ndarray): matrix with shape (n_samples, n_features)
            option      (str): normalizing option

        Returns:
            np.ndarray
        """
        assert isinstance(data, np.ndarray)
        assert option in ('l1', 'l2', 'max')

        return normalize(data, option, axis=1)

    @classmethod
    def l1_norm(cls, data):
        return cls.lp_norm(data, 'l1')

    @classmethod
    def l2_norm(cls, data):
        return cls.lp_norm(data, 'l2')

    @classmethod
    def max_norm(cls, data):
        return cls.lp_norm(data, 'max')

    @classmethod
    def scaler(cls, scaler_cls, data, fitted_scaler=None):
        """
        Performs featue scaling and returns the scaled data

        Args:
            scaler        (MinMaxScaler, StandardScaler): scaler to use
            data                            (np.ndarray): matrix with shape (n_samples, n_features)
            fitted_scaler (MinMaxScaler, StandardScaler): fitted scaler to transform teh data

        Returns:
            np.ndarray
        """
        allowed_scalers = (MinMaxScaler, StandardScaler)
        assert scaler_cls in allowed_scalers
        assert isinstance(data, np.ndarray)

        if fitted_scaler is not None:
            assert isinstance(fitted_scaler, allowed_scalers)
            scaler = fitted_scaler
        else:
            scaler = scaler_cls()
            scaler.fit(data)

        return scaler, scaler.transform(data)

    @classmethod
    def standardize(cls, data, fitted_scaler=None):
        return cls.scaler(StandardScaler, data, fitted_scaler)

    @classmethod
    def normalize(cls, data, fitted_scaler=None):
        return cls.scaler(MinMaxScaler, data, fitted_scaler)

    @classmethod
    def get_normalizer(cls, option, **kwargs):
        """
        Gets the method based on the provided option, calls it using the provided kwargs,
        then returns the results

        Args:
            option (Normalizer.CHOICES): one of the implemented options

        Kwargs:
            keyword arguments for the method called

        Returns:
            Results from called method (could be the normalized data or scaler, normalized_data)
        """
        assert option in cls.CHOICES

        return getattr(cls, option)(**kwargs)
