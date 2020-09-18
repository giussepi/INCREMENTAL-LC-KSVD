# INCREMENTAL LC-KSVD

Python implementation of the Incremental Label consistent KSVD algorithm proposed by Zhuolin Jiang, Zhe Lin and Larry S. Davis.

This implementation is a translation of the matlab code released by the authors
on [http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html](http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html).

The results on Caltech101 using 30 training samples are consistent with those reported by the [authors](http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html)

| Best recognition rate (accuracy) | Iteration |
|:-----------------------:|:-----------:|
|     0.710552268244576 |         3 |
|    0.7140039447731755 |         0 |
|    0.7284681130834977 |         1 |
|    0.7218934911242604 |         1 |
|    0.7345496383957922 |         1 |
|     0.735207100591716 |         0 |
|    0.7355358316896778 |         1 |
|    0.7373438527284681 |         0 |

## INSTALLATION

1. Create a virtual environment and activate it [optional]

2. Install the requirements

	`pip install -r requirements.txt`

3. You can install ATLAS or Intel MKL to work with SPAMS. The [documentation](http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams003.html)
   suggests using MKL to get the best performance.

	1. Atlas:

		1. Run [this installation
    script](https://gist.github.com/giussepi/a1690eb51cc65b7b9cb534d108cc7898). Look
    at [ATLAS stable
    releases](https://sourceforge.net/projects/math-atlas/files/Stable/) and
    [LaPack releses](http://www.netlib.org/lapack/) to update the script if
    necessary.
		2. Install spams

			`pip installs spams`

	2. Intel Math Kernel Library (MKL)

		1. MKL is part of the Intel® Parallel Studio XE Cluster Edition for
    Linux. So you need to register on [its web
    site](https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe/choose-download.html). Scroll
    down and go to the section that says "Get Your Free Software" and choose the
    option that applied to you. After the registration process you will receive
    a serial number and a download link. You must store them on a safe
    place. Finally, just download the installer and follow the steps. Finally,
    test your installation by running (the path could be a bit different):

			`source /opt/intel/parallel_studio_xe_2020/psxevars.sh`

		2. If using a virtual environment then **the above line must be executed**
           **always after activating the environment**. If you are using virtualenvwrapper
           you can place it at `<my virtualenv path>/bin/postactivate` file.

		3. Install spams for MKL

			`pip install spams-mkl`


## Development

1. For some unknown reason numpy must be imported always before spams to avoid
   the importation error `_spams_wrap.cpython-36m-x86_64-linux-gnu.so: undefined symbol: slasrt_`

   ``` python
   import numpy as np
   import spams
   ```
2. Install all your dependencies before running `source /opt/intel/parallel_studio_xe_2020/psxevars.sh`. Otherwise, you could overwrite some Intel programs and cause error like `ImportError: /home/giussepi/.local/lib/python3.7/site-packages/_spams_wrap.cpython-37m-x86_64-linux-gnu.so: undefined symbol: slasrt_`. If this happens to you, you have a good chance of fixing it by unsintalling all your dependencies by running:

	``` bash
	pip uninstall -r requirements.txt
	```
3. All the code must be called from the `main.py` file.

## Dataset handlers

Classes implemented to manage datasets and provide the training and testing data.

### spatialpyramidfeatures4caltech101

Requires downloading the [caltech101 extracted spatial pyramid features](http://www.umiacs.umd.edu/~zhuolin/LCKSVD/features/spatialpyramidfeatures4caltech101.zip)
and placing it in the TRAINING_DATA_DIRECTORY or create a symbolic link to its location. e.g.:

``` bash
cd ~/Downloads
unzip spatialpyramidfeatures4caltech101.zip
cd <path_to_my_project>
mkdir trainingdata
cd trainingdata
ln -s /home/<myuser>/Downloads/spatialpyramidfeatures4caltech101 spatialpyramidfeatures4caltech101
```

Usage

``` python
from utils.datasets.spatialpyramidfeatures4caltech101 import DBhandler

train_feats, train_labels, test_feats, test_labels = DBhandler()()
```

## ILC-KSVD

``` python
from models.ilc_ksvd import ILCksvd
from utils.datasets.spatialpyramidfeatures4caltech101 import DBhandler

ilc_ksvd = ILCksvd(DBhandler)
ilc_ksvd.train()
ilc_ksvd.test(plot=True)
```
Achieving the same results reported in the original paper [Label Consistent
K-SVD: Learning a Discriminative Dictionary for
Recognition](https://ieeexplore.ieee.org/abstract/document/6516503) using
spams + MKL required training the algorithm and then using another python execution
to test it. We assume that this particular behaviour is related with spams and
MKL; because the results, after running the algorithm several times during the
day, are not as good as those obtained after restarting the computer.
