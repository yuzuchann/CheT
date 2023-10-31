# CheT
Chemistry Transformer: Chemistry is the natural language of materials.

## download
Download OQMD dataset by RESTfulAPI.  
Materials_Name, Space_Group, Band_Gap and Stability should be obtained form the dataset.  
Totally 1022603 samples. And 1013412 samples after cleaning.  

## pretrain
CheT shares a similar network framework with BERT, so their hyperparameters have similar settings and effects, including but not limited to the number of encoder layers, embedding length, number of attention headers, batchsize, etc.  
This step can be considered as multitask learning. The parameters of the encoder section will be used in the subsequent steps.  
