# INCREMENTAL-LC-KSVD

## INSTALLATION

1. Create a virtual environment and activate it [optional]

2. You can install ATLAS or Intel MKL to work with SPAMS. The [documentation](http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams003.html)
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

	2. Intel Intel Math Kernel Library (MKL)

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


3. Install the requirements

	`pip install -r requirements.txt`


## Development

1. For some unknown reason numpy must be imported always before spams to avoid
   the importation error `_spams_wrap.cpython-36m-x86_64-linux-gnu.so: undefined symbol: slasrt_`

   ```
   import numpy as np
   import spams
   ```
