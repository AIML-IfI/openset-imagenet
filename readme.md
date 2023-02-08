# Open set on ImageNet
Implementation of the experiments performed in Large-Scale Open-Set Classification Protocols for ImageNet, which has been accepted for publication in WACV 2023.
You can find a [pre-print of the paper including our supplemental material on arXiv](https://arxiv.org/abs/2210.06789).
If you make use of our evaluation protocols or this implementation, please cite the following paper:

    @inproceedings{palechor2023openset,
        author       = {Palechor Anacona, Jesus Andres and Bhoumik, Annesha and G\"unther, Manuel},
        booktitle    = {Winter Conference on Applications of Computer Vision (WACV)},
        title        = {Large-Scale Open-Set Classification Protocols for {ImageNet}},
        year         = {2023},
        organization = {IEEE}
    }

## LICENSE
This code package is open-source based on the BSD license.
Please see `LICENSE` for details.

## Data

All scripts rely on the ImageNet dataset using the ILSVRC 2012 data.
If you do not have a copy yet, it can be downloaded from Kaggle (untested): https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview.
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
It relies on the same configuration files from the `config/` directory where some parts are modified during execution, and read the config files with the names according to the `--algorithms`.
You can run:

    train_imagenet_all.py -g [list-of-gpus]

where `[config]` is the directory containing all configuration files, which is by default `config/`.
You can also select some of the `--protocols` to run on, as well as some of the `--loss-functions`, and some of the `--algorithms`.
The `-g` option can take several GPU indexes, and trainings will be executed in parallel if more than one GPU index is specified.
In case the training stops early for unknown reasons, you can safely use the `--continue` option to continue training from the last epoch -- this option also works for the PROSER training.

When you have a single GPU available, start the script and book a trip to Hawaii, results will finish in about a week.
The more GPUs you can spare, the faster the training will end.
However, make sure that the `threshold` algorithm is always executed first, maybe by running:

    train_imagenet_all.py -g [list-of-gpus] --algorithms threshold
    train_imagenet_all.py -g [list-of-gpus] --algorithms openmax evm proser

### Parameter Optimization

Some of our algorithms will require to adapt the parameters to the different loss functions and protocols.
Particularly, EVM and OpenMax have a set of parameters that should be optimized.
Due to the nature of the algorithms, the `train_imagenet_all.py` script has already trained and saved all parameter combinations as provided in the configuration files of these two algorithms, here the task is only to evaluate the algorithms on unseen data.
Naturally, we will make use of the known and the negative samples of the validation set to perform the parameter optimization.

The parameter optimization will be done via the `parameter_optimization.py` script.
It will read the configuration files of the EVM and OpenMax algorithms, load the images from the validation set, extract features with all trained base networks (as given by the `--losses` parameter), and evaluate the different parameter settings of the algorithms.
Particularly, the CCR values at various FPR thresholds will be computed.
Depending on the protocol, this might require several minutes to hours.
Finally, it will write a separate LaTeX table file per protocol/algorithm/loss combination, and summary LaTeX tables including the best parameters for each algorithm.

.. note::
   Note that also the PROSER algorithm has parameters which we might want to optimize.
   However, since this would require a complete network finetuning for each parameter/protocol/algorithm combination, we do not include PROSER in this script.

The optimized parameters should also be transferred into the `config/test.yaml`, we have done this already for you.

### Evaluation

In order to evaluate all models on the test sets, you can make use of the `evaluate_imagenet.py` script.
This script will use all trained models (as resulting from the `train_imagenet_all.py` script) and extract the features, logits and scores for the test set.
A detailed list of algorithms and parameters is read from the `config/test.yaml` file, which is the default for the `--configuration` option of the `evaluate_imagenet.py` script.
Any model that has not been trained will automatically be skipped.
Otherwise, you can restrict the numbers of `--losses` and `--algorithms`, as well as selecting single `--protocols`.
It is also recommended to run feature extraction on a `--gpu`.
For more options and details on the options, please refer to:

    evaluate_imagenet.py --help

### Plotting

Finally, the `plot_imagenet.py` script can be used to create the plots and result tables from the test set as we have them in the paper.
The script will take information from the same `config/test.yaml` configuration file and make use of all results are generated by the `evaluate_imagenet.py` script.
It will plot all results into a single PDF file (`Results_last.pdf` by default), containing multiple pages.
Page 1 will display all OSCR plots for all algorithms applied to networks trained with all loss functions, where both negative and unknown samples are evaluated for each of the three protocols.
The following three pages will contain score distribution plots of the different algorithms (excluding MaxLogits), separated for the three loss functions.

Again, results that do not exist are skipped automatically.
Since the list of algorithms and loss functions will make the plot very busy, you can try to sub-select several `--losses`, `--algorithms`, or `--protocols` to reduce the number of lines in the plots.

You can also modify other parameters, see:

    plot_imagenet.py --help

Additionally, the script will produce three tables, one for each protocol, where the CCR values at various FPR values are tabularized, for an easier comparison and reference.

## Getting help

In case of trouble, feel free to contact us under [siebenkopf@googlemail.com](mailto:siebenkopf@googlemail.com?subject=Open-Set%20ImageNet%20Protocols)
