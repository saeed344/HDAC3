# HDAC3
# Improved prediction of HDAC3 inhibitors, with deep learning models
In this project, we developed deep learning methods to predict HDAC3 inhibitors.

## File description
### Files in /datasets:
active.fa: Fasta file of non-Nucleic acid-binding proteins. \
non-active.fa: Fasta file of double-stranded DNA binding proteins. \

### Files in /model:
CNN_model.h5: 
CNN_model.jason: 

### Dependencies
pip install Tensorflow/
pip install keras
pip install scikit-learn
pip install xgboost

### Step 1: 
Prepare an inhibitor that need to be predicted in .fasta format.
### Step 2: 

### Step 3: 
python test_model.py 
### Step 4: 
Output: \
The prediction results are summarized in the file "prediction.csv".



