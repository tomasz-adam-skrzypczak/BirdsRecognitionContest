# UJ SN2019 Second Kaggle Competitions

It's my solution for Kaggle Competition - https://www.kaggle.com/c/ujnn2019-2/. Data can be obtain 

## Data Preprocessing and data augmentation
- Raw 10 seconds recording was dividing into 1-second intervals
- Raw audio was transformed to Log-Mel spectrograms
- Only intervals with birds sound were augmented
- Data was augmented online

##  Model and training
- Resnet with linear layer + linear layers at the end
- Because of unbalanced data (more intervals without birds sound), I used weighted cross-entropy
- For final submission was used ensemble of 15 models
