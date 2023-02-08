"""Set of utility functions to produce evaluation figures and histograms."""

from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib import colors
import matplotlib.cm as cm

import yaml

class NameSpace:
    def __init__(self, config):
        # recurse through config
        config = {name : NameSpace(value) if isinstance(value, dict) else value for name, value in config.items()}
        self.__dict__.update(config)

    def __repr__(self):
        return "\n".join(k+": " + str(v) for k,v in vars(self).items())

    def __getitem__(self, key):
        # get nested NameSpace by key
        return vars(self)[key]

    def dump(self, indent=4):
        return yaml.dump(self.dict(), indent=indent)

    def dict(self):
        return {k: v.dict() if isinstance(v, NameSpace) else v for k,v in vars(self).items()}

def load_yaml(yaml_file):
    """Loads a YAML file into a nested namespace object"""
    config = yaml.safe_load(open(yaml_file, 'r'))
    return NameSpace(config)



def dataset_info(protocol_data_dir):
    """ Produces data frame with basic info about the dataset. The data dir must contain train.csv, validation.csv
    and test.csv, that list the samples for each split.
    Args:
        protocol_data_dir: Data directory.
    Returns:
        data frame: contains the basic information of the dataset.
    """
    data_dir = Path(protocol_data_dir)
    files = {'train': data_dir / 'train.csv', 'val': data_dir / 'validation.csv',
             'test': data_dir / 'test.csv'}
    pd.options.display.float_format = '{:.1f}%'.format
    data = []
    for split, path in files.items():
        df = pd.read_csv(path, header=None)
        size = len(df)
        kn_size = (df[1] >= 0).sum()
        kn_ratio = 100 * kn_size / len(df)
        kn_unk_size = (df[1] == -1).sum()
        kn_unk_ratio = 100 * kn_unk_size / len(df)
        unk_unk_size = (df[1] == -2).sum()
        unk_unk_ratio = 100 * unk_unk_size / len(df)
        num_classes = len(df[1].unique())
        row = (split, num_classes, size, kn_size, kn_ratio, kn_unk_size,
               kn_unk_ratio, unk_unk_size, unk_unk_ratio)
        data.append(row)
    info = pd.DataFrame(data, columns=['split', 'classes', 'size', 'kn size', 'kn (%)', 'kn_unk size',
                                       'kn_unk (%)', 'unk_unk size', 'unk_unk (%)'])
    return info


def read_array_list(file_names):
    """ Loads npz saved arrays
    Args:
        file_names: dictionary or list of arrays
    Returns:
        Dictionary of arrays containing logits, scores, target label and features norms.
    """
    list_paths = file_names
    arrays = defaultdict(dict)

    if isinstance(file_names, dict):
        for key, file in file_names.items():
            arrays[key] = np.load(file)
    else:
        for file in list_paths:
            file = str(file)
            name = file.split('/')[-1][:-8]
            arrays[name] = np.load(file)
    return arrays


def calculate_oscr(gt, scores, unk_label=-1):
    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = gt.astype(int)
    kn = gt >= 0
    unk = gt == unk_label

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr = [], []
    # get predicted class for known samples
    pred_class = np.argmax(scores, axis=1)[kn]
    correctly_predicted = pred_class == gt[kn]
    target_score = scores[kn][range(kn.sum()), gt[kn]]

    # get maximum scores for unknown samples
    max_score = np.max(scores, axis=1)[unk]

    # Any max score can be a threshold
    thresholds = np.unique(max_score)

    #print(target_score) #HB
    for tau in thresholds:
        # compute CCR value
        val = (correctly_predicted & (target_score >= tau)).sum() / total_kn
        ccr.append(val)

        val = (max_score >= tau).sum() / total_unk
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    return ccr, fpr


def ccr_at_fpr(gt, scores, fpr_values, unk_label=-1):

    # compute ccr and fpr values from scores
    ccr, fpr = calculate_oscr(gt, scores, unk_label)

    ccrs = []
    for t in fpr_values:
        # get the FPR value that is closest, but above the current threshold
        candidates = np.nonzero(np.maximum(t - fpr, 0))[0]
        if candidates.size > 0:
            ccrs.append(ccr[candidates[0]])
        else:
            ccrs.append(None)

    return ccrs


# get distinguishable colors
import matplotlib.cm
colors = matplotlib.cm.tab10(range(10))

COLORS = {
    "threshold": colors[0],
    "openmax": colors[8],
    "proser": colors[2],
    "evm": colors[3],
    "maxlogits": colors[5]
}

STYLES = {
    "entropic": "dashed",
    "softmax": "solid",
    "garbage": "dotted",
    "p1": "dashed",
    "p2": "dotted",
    "p3": "solid"
}

NAMES = {
    "threshold": "Threshold",
    "openmax": "OpenMax",
    "proser": "PROSER",
    "evm": "EVM",
    "maxlogits": "MaxLogits",
    "entropic": "EOS",
    "softmax": "Softmax",
    "garbage": "Garbage",
    "p1": "P_1",
    "p2": "P_2",
    "p3": "P_3",
    1: "$P_1$",
    2: "$P_2$",
    3: "$P_3$"
}

def plot_single_oscr(fpr, ccr, ax, loss, algorithm, scale):
    linewidth = 1.1
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Manual limits
        ax.set_ylim(0.09, 1)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=100))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif scale == 'semilog':
        ax.set_xscale('log')
        # Manual limits
        ax.set_ylim(0.0, .8)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # MaxNLocator(7))  #, prune='lower'))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_ylim(0.0, 0.8)
        # ax.set_xlim(None, None)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # , prune='lower'))
    # Remove fpr=0 since it cause errors with different ccrs and logscale.
#    if len(x):
#        non_zero = x != 0
#        x = x[non_zero]
#        y = y[non_zero]
    ax.plot(fpr,
            ccr,
            linestyle=STYLES[loss],
            color=COLORS[algorithm],
            linewidth=linewidth)  # marker='2', markersize=1
    return ax


def plot_oscr(arrays, gt, scale='linear', title=None, ax_label_font=13, ax=None, unk_label=-1,):
    """Plots OSCR curves for all given scores.
    The scores are stored as arrays: Float array of dim [N_samples, N_classes].
    The arrays contain scores for various loss functions and algorithms as arrays[loss][algorithm].
    """

    for loss, loss_arrays in arrays.items():
        for algorithm, scores in loss_arrays.items():
            ccr, fpr = calculate_oscr(gt, scores, unk_label)
            ax = plot_single_oscr(fpr, ccr,
                              ax=ax,
                              loss=loss,
                              algorithm=algorithm,
                              scale=scale)
    if title is not None:
        ax.set_title(title, fontsize=ax_label_font)
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False, labelsize=ax_label_font)

    return ax

def oscr_legend(losses, algorithms, figure, **kwargs):
    """Creates a legend with the different line style and colors"""
    # add dummy plots for the different styles
    from matplotlib.lines import Line2D

    # create legend elements
    empty_legend = Line2D([None], [None], marker=".", visible=False)
    padding = len(algorithms) - len(losses)
    a_padding = max(-padding,0)
    l_padding = max(padding, 0)

    # add legend elements with sufficient padding
    legend_elements = \
            [empty_legend]*(l_padding//2) + \
            [Line2D([None], [None], linestyle=STYLES[loss], color="k") for loss in losses] + \
            [empty_legend]*(l_padding//2 + l_padding%2) + \
            [empty_legend]*(a_padding//2) + \
            [Line2D([None], [None], linestyle="solid", color=COLORS[algorithm]) for algorithm in algorithms] + \
            [empty_legend]*(a_padding//2 + + a_padding%2)

    labels = \
            [""] *(l_padding//2) + \
            [NAMES[loss] for loss in losses] + \
            [""]*(l_padding//2 + l_padding%2) + \
            [""] *(a_padding//2) + \
            [NAMES[algorithm] for algorithm in algorithms] + \
            [""]*(a_padding//2 + + a_padding%2)

    # re-order row-first to column-first
    columns = max(len(algorithms), len(losses))

    indexes = [i for j in range(columns) for i in (j, j+columns)]
    legend_elements = [legend_elements[index] for index in indexes]
    labels = [labels[index] for index in indexes]

    figure.legend(handles=legend_elements, labels=labels, loc="lower center", ncol=columns, **kwargs)



def get_histogram(scores,
                  gt,
                  bins=100,
                  log_space=False,
                  geomspace_limits=(1, 1e2)):
    """Calculates histograms of scores"""
    known = gt >= 0
    unknown = gt == -2
    negative = gt == -1

    knowns = scores[known, gt[known]]
    unknowns = np.amax(scores[unknown], axis=1)
    negatives = np.amax(scores[negative], axis=1)

    if log_space:
        lower, upper = geomspace_limits
        bins = np.geomspace(lower, upper, num=bins)
#    else:
#        bins = np.linspace(0, 1, num=bins+1)
    histograms = {}
    histograms["known"] = np.histogram(knowns, bins=bins)
    histograms["unknown"] = np.histogram(unknowns, bins=bins)
    histograms["negative"] = np.histogram(negatives, bins=bins)
    return histograms
