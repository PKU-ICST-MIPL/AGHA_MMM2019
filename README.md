# Introduction
This is the source code of our MMM 2019 paper "Hierarchical Vision-Language Alignment for Video Captioning". Please cite the following paper if you use our code.

Junchao Zhang and Yuxin Peng, "Hierarchical Vision-Language Alignment for Video Captioning", 25th International Conference on MultiMedia Modeling (MMM), pp. 42-54, 2019.

# Dependency
This code is implemented on Torch7 with GPUs.

# Data Preparation
Here we take MSVD (Microsoft video description corpus) dataset as an example. For each video, 30 frames are extracted. For each
video frame, the global feature is extracted by the pre-trained GoogLeNet pool5/7x7 s1 layer. The object-specific, relation-specific and region-specific features are extracted from the [MSDN](https://github.com/yikang-li/MSDN) model.

# Usage
Start training and tesing by executiving the following commands. This will train and test the model on MSVD datatset. 

    - sh train_object.sh  ## train the model of object stream
    - sh train_relation.sh  ## train the model of relation stream
    - sh train_region.sh  ## train the model of region stream
    
    - sh test_single.sh ## test the model of three single stream; it is to test the object stream in default, please edit the script to test the other two models
    - th test_model_ensemble.lua ## fusion the models of three streams
    
For more information, please refer to our [MMM paper](http://link.springer.com/chapter/10.1007/978-3-030-05710-7_4).

Welcome to our [Laboratory Homepage](http://www.wict.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.

# Related repositories

[DMRM](https://ziweiyang.github.io/)