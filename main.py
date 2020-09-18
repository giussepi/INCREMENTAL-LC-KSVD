# -*- coding: utf-8 -*-
""" main """

from models.ilc_ksvd import ILCksvd
# from utils.datasets.spatialpyramidfeatures4caltech101 import DBhandler
from utils.datasets.patchcamelyon import DBhandler


def main():
    ilc_ksvd = ILCksvd(DBhandler)
    # ilc_ksvd.train()
    ilc_ksvd.test(plot=True)


if __name__ == '__main__':
    main()
