# -*- coding: utf-8 -*-
""" utils/utils """

import numpy as np


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
