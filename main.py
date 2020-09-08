# -*- coding: utf-8 -*-
""" main """

from models.ilc_ksvd import ILCksvd
from utils.datasets.spatialpyramidfeatures4caltech101 import DBhandler


def main():
    ilc_ksvd = ILCksvd(DBhandler)
    # ilc_ksvd.train()
    ilc_ksvd.test(plot=True)
    # Best recognition rate for OnlineDL is 0.710552268244576 at iteration 3
    # Best recognition rate for OnlineDL is 0.7140039447731755 at iteration 0
    # Best recognition rate for OnlineDL is 0.7284681130834977 at iteration 1
    # Best recognition rate for OnlineDL is 0.7218934911242604 at iteration 1
    # Best recognition rate for OnlineDL is 0.7345496383957922 at iteration 1
    # Best recognition rate for OnlineDL is 0.735207100591716 at iteration 0
    # Best recognition rate for OnlineDL is 0.7355358316896778 at iteration 1
    # Best recognition rate for OnlineDL is 0.7373438527284681 at iteration 0
    # the paper report for LC-KSVD1 and LC-KSVD2 73.4 and 73.6 respectively for 30 samples
    # per class
    # all the evaluations were done using spatial pyramid features
    # They used 510, 1,020, 1,530, 2,040, 2,550, and 3,060
    # dictionary items/sizes, respectively, for 5, 10, 15, 20, 25, and
    # 30 training samples per category.


if __name__ == '__main__':
    main()
