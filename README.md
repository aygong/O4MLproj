# CS-439 Optimization for Machine Learning



## Group Members：

- Aoyu Gong (aoyu.gong@epfl.ch)
- Sijia Du (sijia.du@epfl.ch)
- Qiyuan Dong(qiyuan.dong@epfl.ch)



## Overview:

This project focuses on a network pruning algorithm proposed on ICLR 2019, namely [SNIP for single-shot network pruning](https://arxiv.org/abs/1810.02340). We reproduced the experiments of the original paper with a similar experimental setup. In particular, we evaluated the algorithm on two datasets with seven network architectures given by the reference. Our implementation is based on the [GitHub repo](https://github.com/namhoonlee/snip-public) provided by the authors of the original paper. To further study its robustness, we validated the effectiveness of the algorithm on an additional dataset and nine network architectures, provided numerical and visualization results with respect to a wide range of configurations, and compared our result with the original paper.



## Requirements:
You can reproduce our best prediction by two ways, using either [Google Colab Pro](https://colab.research.google.com/signup) or your local system.

- Reproduce our result on Google Colab
  - Reproduce our result on Google Colab is very easy with our provided `.ipynb`

  <!-- - Google Colab Pro provides NVIDIA Tesla P100 with 25.46GB RAM. -->

  - One thing to note is that our codes requires Tensorflow 1.14.0, which has some compatibility issue with today's Colab. CUDA and some other dependencies need to be reinstalled accordingly to reproduce the result using GPU. Therefore, it is recommended to reproduce our result with CPU on the Colab. The training time is not intolerably long but the setup process is much simpler.

- Reproduce our result on your local system.
  - To reproduce our result, you should install the following packages:
    - tensorflow-gpu 1.14.0
    - gast 0.2.2


<!-- 
## Models:

The base architecture used in our experiment is LeNet-300 and LeNet-5, which were presented in [our report](./report.pdf).

The details of there models can be found here:
- LeNet-300
- LeNet-5

Moreover, the following models are used for comparisons:

- UNet: [U-net: Convolutional networks for biomedical image segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- DLinkNet: [Linknet: Exploiting encoder representations for efficient semantic segmentation](https://ieeexplore.ieee.org/abstract/document/8305148)
- LinkNet: [D-linknet: Linknet with pretrained encoder and dilated convolution for high resolution satellite imagery road extraction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html)
 -->



<!-- ## Data Processing:
### Training Data Processing:
To adapt to the architecture of the proposed network, we first resize training images and their corresponding masks, denoted by training pairs. Let $H \times W$ denote the resolution of training pairs. Next, let $H' \times W'$ denote the resolution of resized training pairs, satisfying $H' = 32 q$ and $W' = 32 k$ where $q, k \in \mathbb{Z}^{+}$. Here, we set $H'$ and $W'$ to 384. For each resized training pair, we implement the following steps.

- Rotation: We rotate each resized training pair by $\theta \in \{0^{\circ}, 45^{\circ}, 90^{\circ}, 135^{\circ}\}$ (counter-clockwise).
- Flipping: For each rotated training pair, we keep the original one, and further implement the horizontal flip and vertical flip.
- Shifting: We shift each flipped training pair randomly. Let $S_0$ and $S_1$ represent the shift along two axes. Both random variables satisfy $S_0 , S_1 \sim U (−16, 16)$.

In all three steps, the pixels outside the boundaries of the processed training pairs are filled by `reflect`. Given that 100 training pairs are used for training, we will have 1200 processed training pairs after the training data processing.



### Testing Data Processing:
Considering the fact that the resolution of testing images and training images are different, we implement the following testing data processing. First, each testing image is also resized. Let $\bar{H} \times \bar{W}$ and $\bar{H}' \times \bar{W}'$ denote the resolution of testing images and resized ones. Here, we set $\bar{H}' = \lfloor \frac{\bar{H} H'}{H} \rfloor$ and $\bar{W}' = \lfloor \frac{\bar{W} W'}{W} \rfloor$. Each resized testing image is divided into four patches with the size of $H' \times W'$, one at each corner. For each patch, its mask is predicted correspondingly. Next, the four masks are merged into one mask with the size of $\bar{H}' \times \bar{W}'$. This mask is further restored to the original resolution, and is used to create a submission file. The procedure for the testing data processing is presented as follows:

![Testing_data_processing](./__pycache__/Testing_data_processing.png)
 -->


## How to reproduce our experiments:

You can reproduce our experiments in two ways, using either [Google Colab](https://colab.research.google.com/signup) or your local machine.

- Reproduce our experiments in Google Colab Pro.
  - First, `git clone` our repo to your local machine and then update the entire folder to your Google Drove. Make sure the structure of the directory is the same as the **Folder Structure** section.
  - Second, click `Change runtime type` in `Runtime`, and select `GPU` as `Hardware accelerator` (after reinstalling CUDA that works with Tensorflow 1.14.0) or `CPU` (for easier setup) and `High-RAM` as `Runtime shape`.
  - Then, you can directly run the provided Jupyter notebooks under the root directory with `Run all`.

- Reproduce our experiments on your local machine.
  - First, install all the packages mentioned in the **Requirement** section.
  - Second, setup your local Jupyter Notebook environment and run the provided Jupyter notebooks under the root directory with `Run all`.


## Folder Structure:

```
Mini-Project
├── README.md
|
├── results                     # All the results from our experiments
|
├── report.pdf                  # Our report
|
├── run.ipynb                   # The Jupyter notebook to run the algorithm and reproduce our results.
|
├── snip/                       # Folder contains all the sources files to run the algorithm
|   ├── train.py
|   ├── test.py 
|   ├── prune.py
|   ├── network.py
|   ├── model.py
|   ├── mnist.py
|   ├── main.py
|   ├── kmnist.py
|   ├── helpers.py
|   ├── dataset.py
|   ├── cifar.py
|   └── augment.py
|
|
├── cifar-10-batches-py/        # Dataset folder for cifar-10
|
├── KMNIST/                     # Dataset folder for KMNIST
|
└── MNIST/                      # Dataset folder for MNIST
