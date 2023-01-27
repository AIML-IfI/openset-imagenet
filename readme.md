# Open set on ImageNet
Implementation of the experiments performed in Large-Scale Open-Set Classification Protocols for ImageNet, which has been presented in WACV 2023.
You can find a [pre-print of the paper including our supplemental material on arXiv](https://arxiv.org/abs/2210.06789), or download the [final version from WACV](https://openaccess.thecvf.com/content/WACV2023/html/Palechor_Large-Scale_Open-Set_Classification_Protocols_for_ImageNet_WACV_2023_paper.html).
If you make use of our evaluation protocols or this implementation, please cite the paper as follows:

    @inproceedings{palechor2023openset,
        author       = {Palechor, Andres and Bhoumik, Annesha and G\"unther, Manuel},
        booktitle    = {Winter Conference on Applications of Computer Vision (WACV)},
        title        = {Large-Scale Open-Set Classification Protocols for {ImageNet}},
        year         = {2023},
        organization = {IEEE/CVF}
    }

## LICENSE
This code package is open-source based on the BSD license.
Please see `LICENSE` for details.

## Data

All scripts rely on the ImageNet dataset using the ILSVRC 2012 data.
If you do not have a copy yet, it can be [downloaded from Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview) (untested).
The protocols rely on the `robustness` library, which in turn relies on some files that have been distributed with the ImageNet dataset some time ago, but they are not available anymore.
With a bit of luck, you can find the files somewhere online:

* imagenet_class_index.json
* wordnet.is_a.txt
* words.txt

If not, you can also rely on the pre-computed protocol files, which can be found in the provided `protocols.zip` file and extracted via:

    unzip protocols.zip

Finally, we have uploaded the [pre-trained models](https://seafile.ifi.uzh.ch/d/af484b369a2d4d13a04f) to allow for an easier comparison to our methods.
Please download them into a directory `experiments` (the default directory in the below scripts) in order to work with them.


## Setup

We provide a conda installation script to install all the dependencies.
Please run:

    conda env create -f environment.yaml

Afterward, activate the environment via:

    conda activate openset-imagenet-comparison

## Scripts

The directory `openset_imagenet/script` includes several scripts, which are automatically installed and runnable.

### Protocols

You can generate the protocol files using the command `imagenet_protocols.py`.
Please refer to its help for details:

    protocols_imagenet.py --help

Basically, you have to provide the original directory for your ImageNet images, and the directory containing the files for the `robustness` library.
The other options should be changed rarely.

### Training of one base model

The training can be performed using the `train_imagenet.py` script.
It relies on a configuration file as can be found in `config/threshold.yaml`.
Please set all parameters as required (the default values are as used in the paper), and run:

    train_imagenet.py [config] [protocol] -g GPU

where `[config]` is the configuration file, `[protocol]` one of the three protocols.
The `-g` option can be used to specify that the training should be performed on the GPU (**highly recommended**), and you can also specify a GPU index in case you have several GPUs at your disposal.

### Running different algorithms using that model

The other algorithms (EVM, OpenMax, PROSER) can be executed with exactly the same `train_imagenet.py` script.
Simply provide another configuration file from the `config/` directory.
Again, you might want to adapt some parameters in those configuration files, but they are all set according to the results in the paper.

.. note::
   Please make sure that you have run the base model training before executing other algorithms.

### Training of all the models with all of the algorithms in the paper

The `train_imagenet_all.py` script provides a shortcut to train a model with three different loss functions on three different protocols.
It relies on the same configuration files from the `config/` directory where some parts are modified during execution.
You can run:

    train_imagenet_all.py --configuration-direcrtory [config] -g [list-of-gpus]

where `[config]` is the directory containing all configuration files, which is by default `config/`.
You can also select some of the `--protocols` to run on, as well as some of the `--loss-functions`, and some of the `--algorithms`.
The `-g` option can take several GPU indexes, and trainings will be executed in parallel if more than one GPU index is specified.
In case the training stops early for unknown reasons, you can safely use the `--continue` option to continue training from the last epoch -- this option also works for the PROSER training.

When you have a single GPU available, start the script and book a trip to Hawaii, results will finish in about a week.
The more GPUs you can spare, the faster the training will end.
However, make sure that the `threshold` algorithm is always executed first, maybe by running:

    train_imagenet_all.py --configuration-direcrtory [config] -g [list-of-gpus] --algorithms threshold
    train_imagenet_all.py --configuration-direcrtory [config] -g [list-of-gpus] --continue


### Evaluation

Finally, the `plot_imagenet.py` script can be used to perform the plots as we have them in the paper.
This script will use all trained models (as resulting from the `train_imagenet_all.py` script), extract the features and scores for the validation and test set, and plots into a single file (`Results_last.pdf` by default), as well as providing the table from the appendix as a LaTeX table (default: `Results_last.tex`)

1. OSCR curves are presented in Figure 2 of the paper, on the validation and test sets.
2. Confidence propagation plots as in Figure 3, for all three loss functions on the validation set.
3. Histograms of softmax scores as in Figure 4 of the paper.

Please specify the `--imagenet-directory` so that the original image files can be found, and select an appropriate GPU index.
You can also modify other parameters, see:

    plot_imagenet.py --help

For example, you can specify that you want to use the best model based on our confidence measure, via `--use-best`.
You can also regenerate the linear OSCR plots via `--linear`.
You can sort the plots by loss so that you can compare across protocols via `--sort-by-loss`.
For the remaining parameters it is recommended to keep the default values to be able to regenerate the plots from the paper.

The list of commands to reproduce all table and figures, including the supplemental material, is:

    plot_imagenet.py --imagenet-directory [YOUR_IMAGENET_PATH] --gpu [gpu_index]
    plot_imagenet.py --imagenet-directory [YOUR_IMAGENET_PATH] --linear
    plot_imagenet.py --imagenet-directory [YOUR_IMAGENET_PATH] --use-best --gpu [gpu_index]
    plot_imagenet.py --imagenet-directory [YOUR_IMAGENET_PATH] --sort-by-loss

## Getting help

In case of trouble, feel free to contact us under [siebenkopf@googlemail.com](mailto:siebenkopf@googlemail.com?subject=Open-Set%20ImageNet%20Protocols)
