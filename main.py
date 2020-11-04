# -*- coding: utf-8 -*-
""" main """

from ilcksvd.models.ilc_ksvd import ILCksvd
from ilcksvd.utils.datasets.spatialpyramidfeatures4caltech101 import DBhandler
# from ilcksvd.utils.datasets.generic import DBhandler
from ilcksvd.utils.utils import Normalizer


def main():
    # train_feats, train_labels, test_feats, test_labels = DBhandler()()

    # print(train_feats.shape)
    # print(train_labels.shape)
    # print(test_feats.shape)
    # print(test_labels.shape)

    ilc_ksvd = ILCksvd(DBhandler, Normalizer.NONE)
    # ilc_ksvd.train()
    ilc_ksvd.test(plot=False)


if __name__ == '__main__':
    main()
