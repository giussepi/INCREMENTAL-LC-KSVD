# -*- coding: utf-8 -*-
""" utils/datasets/test/test_exceptions """

import unittest

from ilcksvd.utils.datasets.exceptions import DatasetZeroElementFound


class TestDatasetZeroElementFound(unittest.TestCase):

    def test_message(self):
        with self.assertRaisesRegex(DatasetZeroElementFound, DatasetZeroElementFound.message):
            raise DatasetZeroElementFound()


if __name__ == '__main__':
    unittest.main()
