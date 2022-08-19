# Open set on Imagenet
Implementation of the experiments performed in Large-Scale Open-Set Classification Protocols for ImageNet.

## Data

All scripts rely on the ImageNet dataset using the ILSVRC 2012 data.
If you do not have a copy yet, it can be downloaded from Kaggle (untested): https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview
The protocols rely on the `robustness` library, which in turn relies on some files that have been distributed with the ImageNet dataset some time ago, but they are not available anymore.
With a bit of luck, you can find the files somewhere online:

* imagenet_class_index.json
* wordnet.is_a.txt
* words.txt

If not, you can also rely on the pre-computed protocol files, which can be found in the provided `protocols.zip` file and extracted via:

    unzip protocols.zip


## Setup

We provide a conda installation script to install all the dependencies.
Please run:

    conda env create -f environment.yaml

Afterward, activate the environment via:

    conda activate openset-imagenet

## Scripts

The directory `openset_imagenet/script` includes several scripts, which are automatically installed and runnable.

### Protocols

You can generate the protocol files using the command `imagenet_protocols.py`.
Please refer to its help for details:

    protocols_imagenet.py --help

Basically, you have to provide the original directory for your ImageNet images, and the directory containing the files for the `robustness` library.
The other options should be changed rarely.

### Training of one model

The training can be performed using the `train_imagenet.py` script.
It relies on a configuration file as can be found in `config/train.yaml`.
Please set all parameters as required (the default values are as used in the paper), and run:

    train_imagenet.py [config] [protocol] -o [outdir] -g GPU

where `[config]` is the configuration file, `[protocol]` one of the three protocols, and `[outdir]` the output directory of the trained model and some logs.
The `-g` option can be used to specify that the training should be performed on the GPU (**highly recommended**), and you can also specify a GPU index in case you have several GPUs at your disposal.

### Training of all the models in the paper

The `train_imagenet_all.py` script provides a shortcut to train a model with three different loss functions on three different protocols.
It relies on the same configuration file (`config/train.yaml`) where some parts are modifed during execution.
You can run:

    train_imagenet_all.py --configuration [config] -g [list-of-gpus]

where `[config]` is the configuration file, which is by default `config/train.yaml`.
You can also select some of the `--protocols` to run on, as well as some of the `--loss-functions`, or change the `--output-directory`.
The `-g` option can take several GPU indexes, and trainings will be executed in parallel if more than one GPU index is specified.
In case the training stops early for unknown reasons, you can safely use the `--continue` option to continue training from the last epoch.

When you have a single GPU available, start the script and book a trip to Hawaii, results will finish in about a week.
The more GPUs you can spare, the faster the training will end.

### Evaluation

Finally, the `plot_imagenet.py` script can be used to perform the plots as we have them in the paper.
This script will use all trained models (as resulting from the `train_imagenet_all.py` script), extract the features and scores for the validation and test set, and plots into a single file (`Results_last.pdf` by default), as well as providing the table from the appendix as a LaTeX table (default: `Results_last.tex`)

1. OSCR curves are presented in Figure 2 of the paper, on the vaidation and test sets.
2. Confidence propagation plots as in Figure 3, for all three loss functions on the validation set.
3. Histograms of softmax scores as in Figure 4 of the paper.

Please specify the `--imagenet-directory` so that the original image files can be found, and select an appropriate GPU index.
You can also modify other parameters, see:

    plot_imagenet.py --help

For example, you can specify that you want to use the best model based on our confidence measure, via `--use-best`.
For the remaining parameters it is recommended to keep the default values to be able to regenerate the plots from the paper.

The list of commands to reprocude all table and figures, including the supplemental material, is:

    plot_imagenet.py --imagenet-directory [YOUR_IMAGENET_PATH] --gpu [GPU_INDEX]
    plot_imagenet.py --imagenet-directory [YOUR_IMAGENET_PATH] --linear
    plot_imagenet.py --imagenet-directory [YOUR_IMAGENET_PATH] --use-best --gpu [GPU_INDEX]

## Getting help

In case of trouble, feel free to contact us under [guenther@ifi.uzh.ch](mailto:guenther@ifi.uzh.ch?subject=Open-Set%20ImageNet%20Protocols)
