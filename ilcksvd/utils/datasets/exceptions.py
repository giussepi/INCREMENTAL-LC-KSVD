# -*- coding: utf-8 -*-
""" utils/datasets/exceptions.py """


class DatasetZeroElementFound(Exception):
    """
    Exception to be raised when zero element (<=1e-6) is found in the dataset
    """

    message = "Zero element found in the dataset"

    def __init__(self):
        super().__init__(self.message)
