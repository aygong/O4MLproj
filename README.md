# CS-439 Optimization for Machine Learning



## Group Members

- Aoyu Gong (aoyu.gong@epfl.ch)
- Sijia Du (sijia.du@epfl.ch)
- Qiyuan Dong(qiyuan.dong@epfl.ch)



## Overview

This project focuses on a network pruning algorithm proposed on ICLR 2019, namely [SNIP for single-shot network pruning](https://arxiv.org/abs/1810.02340). We reproduced the experiments of the original paper with a similar experimental setup. In particular, we evaluated the algorithm on two datasets with seven network architectures given by the reference. Our implementation is based on the [GitHub repo](https://github.com/namhoonlee/snip-public) provided by the authors of the original paper. To further study its robustness, we validated the effectiveness of the algorithm on an additional dataset and nine network architectures, provided numerical and visualization results with respect to a wide range of configurations, and compared our result with the original paper.



## Requirements
You can reproduce our best prediction by two ways, using either [Google Colab Pro](https://colab.research.google.com/signup) or your local system.

- Reproduce our result on Google Colab
  - Reproduce our result on Google Colab is very easy with our provided `run.ipynb`

  <!-- - Google Colab Pro provides NVIDIA Tesla P100 with 25.46GB RAM. -->

  - One thing to note is that our codes requires Tensorflow 1.14.0, which has some compatibility issue with today's Colab. Be sure to run `%tensorflow_version 1.x` at the beginning of the Colab notebook to setup the environment correctly.

- Reproduce our result on your local system.
  - To reproduce our result, you should install the following packages:
    - tensorflow-gpu 1.14.0
    - gast 0.2.2


## How to reproduce our experiments

You can reproduce our experiments in two ways, using either [Google Colab](https://colab.research.google.com/signup) or your local machine.

- Reproduce our experiments in Google Colab Pro.
  - First, `git clone` our repo to your local machine and then update the entire folder to your Google Drive. Make sure the structure of the directory is the same as the **Folder Structure** section.
  - Second, click `Change runtime type` under the menu `Runtime`, and select `GPU` as `Hardware accelerator` and `High-RAM` as `Runtime shape`.
  - Then, you can just run the provided Jupyter notebooks `run.ipynb` under the root directory with `Run all`.

- Reproduce our experiments on your local machine.
  - First, install all the dependencies mentioned in the **Requirement** section.
  - Second, setup your local Jupyter Notebook environment and run the provided Jupyter notebooks `run.ipynb` under the root directory with `Run all`.


## Folder Structure

```
Mini-Project
├── README.md
|
├── results.xlsx                # All the results from our experiments
|
├── report.pdf                  # Our report
|
├── run.ipynb                   # The Jupyter notebook to run the algorithm and reproduce our results.
|
├── snip/                       # Folder contains all the sources files to run the algorithm
|   ├── train.py                # Defines each training iteration
|   ├── test.py                 # Defines the test process
|   ├── prune.py                # Defines the pruning process which is executed at the beginning of training
|   ├── network.py              # Defines several neural network architectures
|   ├── model.py                # Defines the entire model, including network, loss, optimization, and input/output processing, etc.
|   ├── mnist.py                # Processes dataset MNIST
|   ├── main.py                 # Main file that parses arguments and initialize the entire training and testing procedure
|   ├── kmnist.py               # Processes dataset KMNIST
|   ├── helpers.py              # Several auxiliary functions
|   ├── dataset.py              # Processes the dataset to feed the model
|   ├── cifar.py                # Processes dataset cifar-10
|   └── augment.py              # Perform data augmentation
|
|
├── cifar-10-batches-py/        # Dataset folder for cifar-10
|
├── KMNIST/                     # Dataset folder for KMNIST
|
└── MNIST/                      # Dataset folder for MNIST
