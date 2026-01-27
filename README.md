# BAGF:Cancer Driver Genes Identification Based on A Deep Learning Framework with Bidirectional Attention and Gate Fusion
We propose a deep learning framework BAGF, which mainly includes these modules as follows. Firstly, two Chebyshev graph convolutional networks are used to learn gene representations from two gene feature matrices. Secondly, to fully capture key intrinsic relationship in gene features, a bidirectional attention mechanism is designed to enhance the capacity of feature representations. Thirdly, to reflect the importance of features and dynamically filter out ineffective features, a gate fusion is proposed to integrate the feature representations. Finally, a multi-layer perceptron classifier is selected to predict cancer driver genes. The overview of BAGF is shown as follows:
<img width="865" height="432" alt="image" src="https://github.com/user-attachments/assets/7a91bb70-7564-44b7-b52c-34e87c426583" />
## Requirements
The operating environment for BAGF uses an RTX 4090D GPU(24 GB) and an 18 vCPU AMD EPYC 9754 128 core processor, Ubuntu 18.04 and includes Python 3.8, PyTorch 1.12.1+cu113, and PyTorch Geometric 2.5.2.
* Python 3.8
* PyTorch 1.12.1+cu113
* PyTorch Geometric 2.5.2
## Reproducibility

