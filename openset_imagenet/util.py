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


def calculate_oscr(gt, scores, unk_label=-1, drop_bg=False):
    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes] or [N_samples, N_classes+1]
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
    pred_class = np.argmax(scores, axis=1)

    if drop_bg:  # if background class drop the last column of scores
        scores = scores[:, :-1]

    max_score = np.max(scores, axis=1)
    target_score = scores[kn][range(kn.sum()), gt[kn]]
    #print(target_score) #HB
    for tau in np.unique(target_score)[:-1]:
        val = ((pred_class[kn] == gt[kn]) & (target_score > tau)).sum() / total_kn
        ccr.append(val)

        val = (unk & (max_score > tau)).sum() / total_unk
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    return ccr, fpr


def plot_single_oscr(x, y, ax, exp_name, color, baseline, scale):
    linestyle = 'solid'
    linewidth = 1.1
    if baseline:  # The baseline is always the first array
        linestyle = 'dashed'
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
    if len(x):
        non_zero = x != 0
        x = x[non_zero]
        y = y[non_zero]
    ax.plot(x,
            y,
            label=exp_name,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth)  # marker='2', markersize=1
    return ax


def plot_oscr(arrays, methods, scale='linear', title=None, ax_label_font=13,
              ax=None, unk_label=-1,):

    color_palette = cm.get_cmap('tab10', 10).colors
    #HB
    print( len(arrays),  len(methods) )
    
    assert len(arrays) == len(methods)

    for idx, array in enumerate(arrays):
        has_bg = methods[idx] == "garbage"
        print(methods[idx])

        if array is None:
            ccr, fpr = [], []
        else:
            gt = array['gt']
            scores = array['scores']

            if has_bg:    # If the loss is BGsoftmax then removes the background class
                scores = scores[:, :-1]
            ccr, fpr = calculate_oscr(gt, scores, unk_label)

        ax = plot_single_oscr(x=fpr, y=ccr,
                              ax=ax, exp_name=methods[idx],
                              color=color_palette[idx], baseline=False,
                              scale=scale)
    if title is not None:
        ax.set_title(title, fontsize=ax_label_font)
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False, labelsize=ax_label_font)
    return ax


def get_histogram(array, unk_label=-1,
                  metric='score',
                  bins=100,
                  drop_bg=False,
                  log_space=False,
                  geomspace_limits=(1, 1e2)):
    """Calculates histograms of scores or norms"""
    score = array['scores']
    if drop_bg:  # if background class drop the last column of scores
        score = score[:, :-1]
    gt = array['gt'].astype(np.int64)
    features = array['features']
    norms = np.linalg.norm(features, axis=1)
    kn = (gt >= 0)
    unk = gt == unk_label
    if metric == 'score':
        kn_metric = score[kn, gt[kn]]
        unk_metric = np.amax(score[unk], axis=1)
    elif metric == 'norm':
        kn_metric = norms[kn]
        unk_metric = norms[unk]
    if log_space:
        lower, upper = geomspace_limits
        bins = np.geomspace(lower, upper, num=bins)
    kn_hist, kn_edges = np.histogram(kn_metric, bins=bins)
    unk_hist, unk_edges = np.histogram(unk_metric, bins=bins)
    return kn_hist, kn_edges, unk_hist, unk_edges

# def plot_oscr_knvsunk(arrays, rows=1, cols=1, figsize=(10, 6)):
#     fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True,
#                            constrained_layout=True)
#     if rows * cols > 1:
#         ax = np.ravel(ax)
#     for idx, exp_name in enumerate(sorted(arrays, reverse=True)):
#         ccr, fpr_db, fpr_da, _ = calculate_oscr(
#             gt=arrays[exp_name]['gt'],
#             scores=arrays[exp_name]['scores'],
#             testing=True
#         )
#         if idx == 0:
#             sccr, sfpra, sname = ccr, fpr_da, exp_name
#         ax[idx].plot(fpr_db, ccr, label=exp_name + '_ku', linewidth=1)
#         ax[idx].plot(fpr_da, ccr, label=exp_name + '_uu', linewidth=1)
#         ax[idx].plot(sfpra, sccr, label=sname + '_suu', linestyle=':', linewidth=1)
#         ax[idx].set_xlim(0, 1)
#         ax[idx].set_ylim(0, 1)
#         ax[idx].xaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
#         ax[idx].tick_params(bottom=True, top=True, left=True, right=True, direction='in')
#         ax[idx].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                             labelsize=10)
#         ax[idx].legend(frameon=False, loc="lower right", fontsize=11)
#     fig.suptitle('OSCR: Known-unknowns and Unknown-unknowns ', fontsize=14)
#     fig.supxlabel("False positivie rate", fontsize=13)
#     fig.supylabel("Correct classification rate", fontsize=13)
#     plt.show()


def get_best_arrays(files_dict):
    best_paths = dict()
    for name, path in files_dict.items():
        exception_list = ['$S_2$', '$E_2$', '$O_2$', '$S_3$', '$E_3$', '$O_3$',
                          '$S_1$', '$E_1$', '$O_1$']
        if name in exception_list:
            best_paths[name] = path
        best_paths[name] = Path(str(path).replace('_curr_', '_best_'))
    return best_paths


# def plot_histogram_val_test(arrays_val, arrays_test, metric, bins, figsize, title, linewidth=1,
#                             split='test', font=14, sharex=True, sharey=True, log=True,
#                             normalized=True):
#     # Create general figure
#     nrows = len(arrays_val)
#     fig, ax = plt.subplots(nrows, 2, figsize=figsize, constrained_layout=True, sharex=sharex,
#                            sharey=sharey)
#     fig.suptitle(title, fontsize=font + 1)
#     # Iterate over experiments
#     for idx, (exp_name, array) in enumerate(arrays_val.items()):
#         plot_single_histogram(exp_name, array, size=(6, 4), value=metric, ax=ax[idx, 0], bins=bins,
#                               legend=False, ax_label_font=font, log=log, linewidth=linewidth,
#                               label1='Knowns', label2='Unknowns', normalized=normalized,
#                               split='val')

#     for idx, (exp_name, array) in enumerate(arrays_test.items()):
#         plot_single_histogram(exp_name, array, size=(6, 4), value=metric, ax=ax[idx, 1], bins=bins,
#                               legend=False, ax_label_font=font, log=log, linewidth=linewidth,
#                               label1='Knowns', label2='Unknowns', normalized=normalized,
#                               split='test')

#     # set custom legend
#     handles = [Line2D([], [], c='tab:blue'), Line2D([], [], c='indianred')]
#     labels = ['KKs', 'UUs']
#     fig.legend(frameon=False, bbox_to_anchor=(0.5, -0.1), loc='lower center',
#                fontsize=font, handles=handles, labels=labels, ncol=2)
#     fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.1, wspace=0.05)


# def plot_single_histogram(arrays, value='score', ax=None, bins=200,
#                           legend=True, ax_label_font=13, scale='linear', linewidth=1,
#                           label1='Knowns', label2='Unknowns', normalized=True, split='test',
#                           xlim=None, linestyle='solid'):

#     score = arrays['scores']
#     gt = arrays['gt'].astype(np.int64)
#     feat = arrays['features']
#     norms = np.linalg.norm(feat, axis=1)
#     kn = (gt >= 0)
#     unk = ~kn
#     if split == 'test':
#         unk = gt == -2

#     kn_scores = score[kn, gt[kn]]
#     unk_scores = np.amax(score[unk], axis=1)
#     kn_norms = norms[kn]
#     unk_norms = norms[unk]

#     kn_metric = kn_scores
#     unk_metric = unk_scores

#     if value == 'product':
#         kn_metric = kn_scores * kn_norms
#         unk_metric = unk_scores * unk_norms
#     elif value == 'norm':
#         kn_metric = kn_norms
#         unk_metric = unk_norms
#     print('kn metrics', np.mean(kn_metric))
#     print('un metrics', np.mean(unk_metric))

#     # Create histograms
#     max_metric = max(np.max(kn_metric), np.max(unk_metric))
#     bins = np.linspace(0, max_metric, bins)
#     # bins_mean = centers = 0.5*(bins[1:]+ bins[:-1])
#     hist_kn, _ = np.histogram(kn_metric, bins)
#     hist_unk, _ = np.histogram(unk_metric, bins)
#     if normalized:
#         # max_val = max(np.max(hist_kn), np.max(hist_unk))
#         hist_kn = hist_kn / np.max(hist_kn)
#         hist_unk = hist_unk / np.max(hist_unk)
#         # hist_kn = hist_kn/max_val
#         # hist_unk = hist_unk/max_val
#         # ax.yaxis.set_major_locator(FixedLocator([0.5, 1]))
#     # Custom plot
#     if xlim is not None:
#         ax.set_xlim(xlim)
#     edge_unk = colors.to_rgba('indianred', 1)
#     fill_unk = colors.to_rgba('firebrick', 0.02)
#     edge_kn = colors.to_rgba('tab:blue', 1)
#     fill_kn = colors.to_rgba('tab:blue', 0.02)
#     ax.stairs(hist_kn, bins, fill=False, color=fill_kn, edgecolor=edge_kn,
#               linewidth=linewidth, linestyle=linestyle)
#     ax.stairs(hist_unk, bins, fill=False, color=fill_unk, edgecolor=edge_unk,
#               linewidth=linewidth, linestyle=linestyle)

#     if scale == 'semilog':
#         ax.set_xscale('log')
#     if scale == 'log':
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         y_major = LogLocator(base=10.0, numticks=5)
#         ax.yaxis.set_major_locator(y_major)
#         y_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=5)
#         ax.yaxis.set_minor_locator(y_minor)
#         ax.yaxis.set_minor_formatter(NullFormatter())

#     ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
#     ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True,
#                    labelright=False, labelsize=ax_label_font)

#     # ax.set_title(exp_name, fontsize=ax_label_font)
#     # legend
#     if legend:
#         handles = [Line2D([], [], c=edge_kn), Line2D([], [], c=edge_unk)]
#         labels = [label1, label2]
#         ax.legend(frameon=False, bbox_to_anchor=(0.5, -0.15), loc='lower center', fontsize=12,
#                   handles=handles, labels=labels, ncol=2)
