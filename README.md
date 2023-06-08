# DeepSolar Extension
Extending Stanford's DeepSolar model to handle distribuition shifts (eg. satellite datasets from other countries)


This repository has multiple branches implementing finetuning strategies. 


- finetune: xx
- lisa: Implements LISA (Learning Invariant Predictors with Selective Augmentation), a data augmentation method that linearly interpolates between training example inputs and labels (Yao, 2022). Within the src/ folder of this branch, there are scripts for hyperparameter tuning/training the segmentation and classification branches of the original DeepSolar model using this method. There are also scripts for evaluating the finetuned and baseline models, and .csv files containing the results reported in the paper. (Reference: Huaxiu Yao, Yu Wang, Sai Li, Linjun Zhang, Weixin Liang, James Zou, and Chelsea Finn. Improving out-of-distribution robustness via selective augmentation, 2022. https://arxiv.org/pdf/2201.00299.pdf)
- gan: xx
