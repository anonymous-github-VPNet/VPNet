# Code for VPNet

## Dataset - SemanticKITTI
Download voxel dataset from the SemanticKITTI website http://semantic-kitti.org

Download point cloud dataset from KITTI website https://www.cvlibs.net/datasets/kitti/

## Dataset - SemanticPOSS
Download SemanticPOSS dataset from http://www.poss.pku.edu.cn/semanticposs.html

## Installation

Install the following core dependencies.

torch

cudatoolkit

torch_scatter

chamfer_distance

spconv

numpy

numba

- We use chamfer_distance as geometry loss of CVP module. Download chamfer_distance and build it in your environment.
- We use torch_scatter to gather the features that belong to the same voxel.
- When installing spconv, download it from the repo of spconv, and build it in your environment.
- While facing env problem, try to switch the spconv version and torch_scatter version.
## Usage

### Model Config

Renew the config in networks/common/model.py

### Training Config
Renew the config in SSA_SC.yaml


### Dataloader
Update the dataloader with the label recitification algorithm defined in https://github.com/SCPNet/Codes-for-SCPNet to improve the completion result as it preprocesses the data to handle the problem caused by dynamic objects during scanning.

### Training
Prepare Lovasz_Softmax function in this folder

networks/common/


Renew the config in train.py and 

python networks/train.py


### Validation
Renew the config and ckpt in validate.py and

python networks/validate.py


### Testing

Renew the config and ckpt in test.py and 

python networks/test.py

### Conv Modules Equipment
Add some convolution modules to the model to improve the performance as https://github.com/SCPNet/Codes-for-SCPNet.

We renew the model in the following way.

1. Add convolution modules after point feature extraction. We define the convolution module in networks/common/ConvCompletion.py
2. Switch voxel features and BEV features to initally completed. 
3. Complete through proposed dual-branch network including CVP module.



