# import torch
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from PIL import Image
from pathlib import Path
from collections import defaultdict
# from sklearn.metrics import roc_auc_score
from matplotlib.ticker import MaxNLocator, FixedLocator
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from matplotlib import colors
# import losses, model


# def get_checkpoint_epoch(file_path):
#     '''Prints the epoch in which a checkpoint was saved'''
#     checkpoint = torch.load(file_path, map_location='cpu')
#     epoch = checkpoint['epoch']
#     del checkpoint
#     return epoch


def dataset_info(protocol_data_dir):
    data_dir = Path(protocol_data_dir)
    files = {'train': data_dir/'train.csv', 'val': data_dir/'validation.csv', 'test': data_dir/'test.csv'}
    pd.options.display.float_format = '{:.1f}%'.format
    data = []
    for split, path in files.items():
        df = pd.read_csv(path, header=None)
        size = len(df)
        kn_size = (df[1] >= 0).sum()
        kn_ratio = 100*kn_size/len(df)
        kn_unk_size = (df[1] == -1).sum()
        kn_unk_ratio = 100*kn_unk_size/len(df)
        unk_unk_size = (df[1] == -2).sum()
        unk_unk_ratio = 100*unk_unk_size/len(df)
        num_classes = len(df[1].unique())
        row = (split, num_classes, size, kn_size, kn_ratio, kn_unk_size,
               kn_unk_ratio, unk_unk_size, unk_unk_ratio)
        data.append(row)
    info = pd.DataFrame(data, columns=['split', 'classes', 'size', 'kn size', 'kn (%)',
                                       'kn_unk size', 'kn_unk (%)', 'unk_unk size', 'unk_unk (%)'])
    return info


def process_names(file_list):
    list_paths = []
    for file in file_list:
        file = str(file)
        name = file.split('_')[:-3]
        name = '_'.join(name)
        path = '/'.join(['runs', name, file])
        list_paths.append(path)
    return list_paths


def read_array_list(file_names, process=False):
    if process:
        list_paths = process_names(file_names)
    else:
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


def get_ccr_at_fpr(gt, scores, tau, split='test'):
    # predictions
    pred_class = np.argmax(scores, axis=1)
    # gt = gt.astype(np.int64)
    pred_score = scores[np.arange(len(scores)), pred_class]
    # pred_score = scores[np.arange(len(scores)), gt]
    max_score = np.amax(scores, axis=1)
    
    # Total number of samples in:
    dc = np.sum(gt > -1)    # Dc = Cardinality of known samples
    db = np.sum(gt == -1)   # Db = Cardinality of known unknown samples
    du = np.sum(gt < 0)     # Du = All unknown samples (Db U Da)
    da = np.sum(gt == -2)   # Da = Cardinality of unknown unknown samples
    corr_rate = np.count_nonzero((gt >= 0)*(pred_class == gt)*(pred_score >= tau))/dc

    if split=='test':
        unk_unk_rate = np.count_nonzero((gt == -2)*(max_score >= tau))/da
        return tau, corr_rate, unk_unk_rate
    else:
        known_unk_rate = np.count_nonzero((gt == -1)*(max_score >= tau))/db
        return tau, corr_rate, known_unk_rate


def calculate_oscr(gt, scores, norms=None, points=1000):
    # predictions
    pred_class = np.argmax(scores, axis=1)
    # gt = gt.astype(np.int64)
    pred_score = scores[np.arange(len(scores)), pred_class]
    # pred_score = scores[np.arange(len(scores)), gt]
    max_score = np.amax(scores, axis=1)

    if norms is not None:   # This create thresholds over the product of norms*scores
        pred_score = pred_score * norms

    # Total number of samples in:
    dc = np.sum(gt > -1)    # Dc = Cardinality of known samples
    db = np.sum(gt == -1)   # Db = Cardinality of known unknown samples
    du = np.sum(gt < 0)     # Du = All unknown samples (Db U Da)
    da = np.sum(gt == -2)   # Da = Cardinality of unknown unknown samples

    ccr, fpr_db, fpr_da, fpr_du = [], [], [], []
    # print(len(np.unique(pred_score)))
    # for tau in np.unique(pred_score):
    for tau in np.linspace(start=0, stop=1, num=points):
        # correct classifiation rate
        corr_rate = np.count_nonzero((gt >= 0)*(pred_class == gt)*(pred_score >= tau))/dc
        ccr.append(corr_rate)

        # false positive rate known unknown
        known_unk_rate = np.count_nonzero((gt == -1)*(max_score >= tau))/db
        fpr_db.append(known_unk_rate)

        # false positive rate all unknowns
        unk_rate = np.count_nonzero((gt < 0)*(max_score >= tau))/du
        fpr_du.append(unk_rate)

        #  false positive rate uses unknown unknown
        if da != 0:
            unk_unk_rate = np.count_nonzero((gt == -2)*(max_score >= tau))/da
            fpr_da.append(unk_unk_rate)
    # if the metric for validation then fpr_Du=fpr_Db
    return ccr, fpr_db, fpr_da, fpr_du


def plot_oscr(arrays, figsize=(10, 6), split='val', scale='linear', ylim=1, use_norms=False, colors=None,
              base_line=True, title=None, linewidth=1, ax_label_font=13, ax=None, marker=None, marker_step=None,
              legend_pos="lower right", points=1000):
    if colors is None:
        # colors = sns.color_palette("deep")
        colors = sns.color_palette("tab10")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
    # fig.tight_layout()
    for idx, exp_name in enumerate(arrays):
        # calculate rates for every model
        norms = None
        if use_norms:
            feat = arrays[exp_name]['features']
            norms = np.linalg.norm(feat, axis=1)

        ccr, fpr_db, fpr_da, fpr_du = calculate_oscr(arrays[exp_name]['gt'],
                                                     arrays[exp_name]['scores'],
                                                     norms, points=points)

        linestyle = 'solid'
        if base_line and idx == 0:  # The baseline is always the first array
            linestyle = 'dashed'
        # print(split)
        if split == 'test':
            fpr = fpr_da
        elif split == 'both':
            fpr = fpr_du
        else:
            fpr = fpr_db
            
        if marker is not None:
            ax.plot(fpr, ccr, label=exp_name, linestyle=linestyle, color=colors[idx], linewidth=linewidth, marker=marker, markevery=marker_step)
        else:
            ax.plot(fpr, ccr, label=exp_name, linestyle=linestyle, color=colors[idx], linewidth=linewidth)
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylim(None, 1)
        # ax.set_xlim(None, 1)
    elif scale == 'semilog':
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_xlim(None, 1)
    else:
        ax.set_ylim(0, ylim)
        ax.set_xlim(None, None)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='lower'))

    # plot parameters:
    if title is not None:
        ax.set_title(title, fontsize=ax_label_font)
    # ax.set_xlabel('FPR', fontsize=ax_label_font)
    # ax.set_ylabel('CCR', fontsize=ax_label_font)
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False, labelsize=ax_label_font)
    if legend_pos == 'box':
        ax.legend(frameon=False, bbox_to_anchor=(0, 0), loc="upper right", ncol=1, fontsize=ax_label_font-1)
    elif legend_pos != 'no':
        ax.legend(frameon=False, loc=legend_pos, fontsize=ax_label_font-1)
    return ax

def plot_oscr_knvsunk(arrays, rows=1, cols=1, figsize=(10, 6)):
    fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
    if rows*cols > 1:
        ax = np.ravel(ax)
    for idx, exp_name in enumerate(sorted(arrays, reverse=True)):
        ccr, fpr_db, fpr_da, _ = calculate_oscr(
            gt=arrays[exp_name]['gt'],
            scores=arrays[exp_name]['scores'],
            testing=True
            )
        if idx == 0:
            sccr, sfpra, sname = ccr, fpr_da, exp_name
        ax[idx].plot(fpr_db, ccr, label=exp_name+'_ku', linewidth=1)
        ax[idx].plot(fpr_da, ccr, label=exp_name+'_uu', linewidth=1)
        ax[idx].plot(sfpra, sccr, label=sname+'_suu', linestyle=':', linewidth=1)
        ax[idx].set_xlim(0, 1)
        ax[idx].set_ylim(0, 1)
        ax[idx].xaxis.set_major_locator(MaxNLocator(prune='lower'))
        ax[idx].tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax[idx].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=10)
        ax[idx].legend(frameon=False, loc="lower right", fontsize=11)
    fig.suptitle('OSCR: Known-unknowns and Unknown-unknowns ', fontsize=14)
    fig.supxlabel("False positivie rate", fontsize=13)
    fig.supylabel("Correct classification rate", fontsize=13)
    plt.show()


def transform_to_test(files_dict):
    test_paths = dict()
    for name, path in files_dict.items():
        test_paths[name] = Path(str(path).replace('_val_', '_test_'))
    return test_paths


def get_best_arrays(files_dict):
    best_paths = dict()
    for name, path in files_dict.items():
        exception_list = ['$S_2$', '$E_2$', '$O_2$', '$S_3$', '$E_3$', '$O_3$', '$S_1$', '$E_1$', '$O_1$']
        if name in exception_list:
            best_paths[name] = path
        best_paths[name] = Path(str(path).replace('_curr_', '_best_'))
    return best_paths


def plot_histogram_val_test(arrays_val, arrays_test, metric, bins, figsize, title, linewidth=1,
                            split='test', font=14, sharex=True, sharey=True, log=True, normalized=True):
    # Create general figure
    nrows = len(arrays_val)
    fig, ax = plt.subplots(nrows, 2, figsize=figsize, constrained_layout=True, sharex=sharex, sharey=sharey)
    fig.suptitle(title, fontsize=font+1)
    # Iterate over experiments
    for idx, (exp_name, array) in enumerate(arrays_val.items()):
        plot_single_histogram(exp_name, array, size=(6, 4), value=metric, ax=ax[idx, 0], bins=bins,
                              legend=False, ax_label_font=font, log=log, linewidth=linewidth, label1='Knowns',
                              label2='Unknowns', normalized=normalized, split='val')

    for idx, (exp_name, array) in enumerate(arrays_test.items()):
        plot_single_histogram(exp_name, array, size=(6, 4), value=metric, ax=ax[idx, 1], bins=bins,
                              legend=False, ax_label_font=font, log=log, linewidth=linewidth, label1='Knowns',
                              label2='Unknowns', normalized=normalized, split='test')

    # set custom legend
    handles = [Line2D([], [], c='tab:blue'), Line2D([], [], c='indianred')]
    labels = ['KKs', 'UUs']
    fig.legend(frameon=False,  bbox_to_anchor=(0.5, -0.1), loc='lower center',
               fontsize=font, handles=handles, labels=labels, ncol=2)
    fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.1, wspace=0.05)


def plot_single_histogram(exp_name, arrays, size=(6, 4), value='score', ax=None, bins=200, legend=True,
                          ax_label_font=13, log=True, linewidth=1, fig_title=None, label1='Knowns',
                          label2='Unknowns', normalized=True, split='test', xlim=None, linestyle='solid'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=size, constrained_layout=True)
        fig.suptitle(fig_title, fontsize=ax_label_font)

    score = arrays['scores']
    gt = arrays['gt'].astype(np.int64)
    feat = arrays['features']
    norms = np.linalg.norm(feat, axis=1)
    kn = (gt >= 0)
    unk = ~kn
    if split == 'test':
        unk = gt == -2

    kn_scores = score[kn, gt[kn]]
    unk_scores = np.amax(score[unk], axis=1)
    kn_norms = norms[kn]
    unk_norms = norms[unk]

    kn_metric = kn_scores
    unk_metric = unk_scores

    if value == 'product':
        kn_metric = kn_scores*kn_norms
        unk_metric = unk_scores*unk_norms
    elif value == 'norm':
        kn_metric = kn_norms
        unk_metric = unk_norms
    print('kn metrics', np.mean(kn_metric))
    print('un metrics', np.mean(unk_metric))

    # Create histograms
    max_metric = max(np.max(kn_metric), np.max(unk_metric))
    bins = np.linspace(0, max_metric, bins)
    # bins_mean = centers = 0.5*(bins[1:]+ bins[:-1])
    hist_kn, _ = np.histogram(kn_metric, bins)
    hist_unk, _ = np.histogram(unk_metric, bins)
    if normalized and not log:
        # max_val = max(np.max(hist_kn), np.max(hist_unk))
        hist_kn = hist_kn/np.max(hist_kn)
        hist_unk = hist_unk/np.max(hist_unk)
        # hist_kn = hist_kn/max_val
        # hist_unk = hist_unk/max_val
        ax.yaxis.set_major_locator(ticker.FixedLocator([0.5, 1]))
    # Custom plot
    if xlim is not None:
        ax.set_xlim(xlim)
    edge_unk = colors.to_rgba('indianred', 1)
    fill_unk = colors.to_rgba('firebrick', 0.02)
    edge_kn = colors.to_rgba('tab:blue', 1)
    fill_kn = colors.to_rgba('tab:blue', 0.02)
    ax.stairs(hist_kn, bins, fill=False, color=fill_kn, edgecolor=edge_kn, linewidth=linewidth, linestyle=linestyle)
    ax.stairs(hist_unk, bins, fill=False, color=fill_unk, edgecolor=edge_unk, linewidth=linewidth, linestyle=linestyle)

    
    if log:
        # ax.set_yscale('log')
        ax.set_xscale('log')
        y_major = LogLocator(base=10.0, numticks=5)
        ax.yaxis.set_major_locator(y_major)
        y_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=5)
        ax.yaxis.set_minor_locator(y_minor)
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False, labelsize=ax_label_font)

    # ax.set_title(exp_name, fontsize=ax_label_font)
    # legend
    if legend:
        handles = [Line2D([], [], c=edge_kn), Line2D([], [], c=edge_unk)]
        labels = [label1, label2]
        ax.legend(frameon=False,  bbox_to_anchor=(0.5, -0.15), loc='lower center', fontsize=12,
                  handles=handles, labels=labels, ncol=2)
