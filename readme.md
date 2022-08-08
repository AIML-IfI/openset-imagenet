# Open set on Imagenet
Implementation of the experiments performed in Large-Scale Open-Set Classification
Protocols for ImageNet. 
## Setup
The application was developed using Python 3.8.13 and the modules listed in the
requirements.txt file. It is recommended to create a new virtual environment to 
avoid dependencies issues. For example:

`conda create --name torch111 python=3.8`

`conda activate torch111`

and then install the requirements

`pip install -r /path/to/requirements.txt`

## Protocols
The script protocols.py creates the imagenet open-set protocols and saves three csv files: test, val and test.
Usage example:

`python protocols.py -p 1 --imagenet_dir /path/to/imagenet --metadata_dir /path/to/metadata --out_dir /path/to/dir --seed 42`

