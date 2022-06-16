.. Dynamic Graph Neural Networks documentation master file, created by
   sphinx-quickstart on Thu May  5 10:55:44 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dynamic Graph Neural Networks's documentation!
=========================================================

.. toctree::
   :maxdepth: 2
   :caption: API References
   :hidden:

   modules


Dynamic Graph Neural Networks (DGNN) is a Python package for building and training
dynamic graph neural networks. 


Installation
------------

We develop the package with the following environments:

- Ubuntu 20.04LTS
- gcc 9.4
- CUDA 11.3
- cmake 3.23
- python 3.8

Python dependencies:

- `pytorch`
- `dgl`

C++ dependencies:

- `rmm` (how_to_install_)

.. _how_to_install: https://github.com/yuchenzhong/cs-notes/blob/main/CUDA/rmm/README.md

.. code:: bash

    git clone --recursive https://github.com/yuchenzhong/dynamic-graph-neural-network.git
    cd dynamic-graph-neural-network
    pip install -e .


Index
-----
* :ref:`genindex`
