# Open set on Imagenet
Implementation of the experiments performed in Large-Scale Open-Set Classification
Protocols for ImageNet. 
## Setup
The modules were developed using Python 3.8.13, Pytorch 1.11 and the libraries listed in
requirements.txt. It is recommended to create a new virtual environment to 
avoid dependencies issues. For example, using miniconda:

`conda create --name torch111 python=3.8`

`conda activate torch111`


`pip install -r /path/to/requirements.txt`

## Protocols
The script protocols.py creates the imagenet open-set protocols and creates three csv files: test, val and test.
Usage example:

```
python protocols.py \
--prot 1 \
--imagenet_dir /path/to/imagenet \
--metadata_dir /path/to/metadata \
--out_dir /path/to/dir \
--seed 42
```

## Train

Command line basic example:
```
CUDA_VISIBLE_DEVICES=1 python src/train.py \
name=experiment1 \
data.train_file = data/v2.0/p1_train_tiny.csv \
data.val_file = data/v2.0/p1_val_tiny.csv \
data.test_file = data/v2.0/p1_test_tiny.csv \
loss.type = entropic \
epochs = 100 \
```