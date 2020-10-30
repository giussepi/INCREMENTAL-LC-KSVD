# -*- coding: utf-8 -*-
""" utils/test/test_utils """

import unittest

import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from utils.utils import Normalizer


class Test_Normalizer(unittest.TestCase):

    def setUp(self):
        samples = 3
        features = 5
        self.data = np.random.rand(samples, features)

    def test_none(self):
        self.assertTrue(np.array_equal(
            self.data,
            Normalizer.none(self.data)
        ))

    def test_l1_norm(self):
        self.assertTrue(np.array_equal(
            normalize(self.data, 'l1'),
            Normalizer.l1_norm(self.data)
        ))

    def test_l2_norm(self):
        self.assertTrue(np.array_equal(
            normalize(self.data, 'l2'),
            Normalizer.l2_norm(self.data)
        ))

    def test_max_norm(self):
        self.assertTrue(np.array_equal(
            normalize(self.data, 'max'),
            Normalizer.max_norm(self.data)
        ))

    def test_standardize(self):
        self.assertTrue(np.array_equal(
            StandardScaler().fit_transform(self.data),
            Normalizer.standardize(self.data)[1]
        ))

    def test_standardize_with_scaler(self):
        scaler = StandardScaler()
        scaler.fit(self.data)
        self.assertTrue(np.array_equal(
            scaler.transform(self.data),
            Normalizer.standardize(self.data, scaler)[1]
        ))
        self.assertEqual(scaler, Normalizer.standardize(self.data, scaler)[0])

    def test_normalize(self):
        self.assertTrue(np.array_equal(
            MinMaxScaler().fit_transform(self.data),
            Normalizer.normalize(self.data)[1]
        ))

    def test_normalize_with_scaler(self):
        scaler = MinMaxScaler()
        scaler.fit(self.data)
        self.assertTrue(np.array_equal(
            scaler.transform(self.data),
            Normalizer.normalize(self.data, scaler)[1]
        ))
        self.assertEqual(scaler, Normalizer.normalize(self.data, scaler)[0])

    def test_get_normalizer_none(self):
        self.assertTrue(np.array_equal(
            Normalizer.none(self.data),
            Normalizer.get_normalizer(Normalizer.NONE, data=self.data)
        ))

    def test_get_normalizer_l1_norm(self):
        self.assertTrue(np.array_equal(
            Normalizer.l1_norm(self.data),
            Normalizer.get_normalizer(Normalizer.L1_NORM, data=self.data)
        ))

    def test_get_normalizer_l2_norm(self):
        self.assertTrue(np.array_equal(
            Normalizer.l2_norm(self.data),
            Normalizer.get_normalizer(Normalizer.L2_NORM, data=self.data)
        ))

    def test_get_normalizer_max_norm(self):
        self.assertTrue(np.array_equal(
            Normalizer.max_norm(self.data),
            Normalizer.get_normalizer(Normalizer.MAX_NORM, data=self.data)
        ))

    def test_get_normalizer_standardize(self):
        self.assertTrue(np.array_equal(
            Normalizer.standardize(self.data)[1],
            Normalizer.get_normalizer(Normalizer.STANDARDIZE, data=self.data)[1]
        ))

    def test_get_normalizer_standardize_with_scaler(self):
        scaler = StandardScaler()
        scaler.fit(self.data)
        self.assertTrue(np.array_equal(
            Normalizer.standardize(self.data, fitted_scaler=scaler)[1],
            Normalizer.get_normalizer(Normalizer.STANDARDIZE, data=self.data, fitted_scaler=scaler)[1]
        ))
        self.assertTrue(
            Normalizer.standardize(self.data, fitted_scaler=scaler)[0],
            Normalizer.get_normalizer(Normalizer.STANDARDIZE, data=self.data, fitted_scaler=scaler)[0]
        )

    def test_get_normalizer_normalize(self):
        self.assertTrue(np.array_equal(
            Normalizer.normalize(self.data)[1],
            Normalizer.get_normalizer(Normalizer.NORMALIZE, data=self.data)[1]
        ))

    def test_get_normalizer_normalize_with_scaler(self):
        scaler = MinMaxScaler()
        scaler.fit(self.data)
        self.assertTrue(np.array_equal(
            Normalizer.normalize(self.data, fitted_scaler=scaler)[1],
            Normalizer.get_normalizer(Normalizer.NORMALIZE, data=self.data, fitted_scaler=scaler)[1]
        ))
        self.assertTrue(
            Normalizer.normalize(self.data, fitted_scaler=scaler)[0],
            Normalizer.get_normalizer(Normalizer.NORMALIZE, data=self.data, fitted_scaler=scaler)[0]
        )

    def test_functor(self):
        self.assertTrue(np.array_equal(
            Normalizer.l1_norm(self.data),
            Normalizer()(Normalizer.L1_NORM, data=self.data)
        ))


if __name__ == '__main__':
    unittest.main()
