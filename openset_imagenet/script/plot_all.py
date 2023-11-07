"""Training of all models for the paper"""

import argparse
import multiprocessing
import collections
import subprocess
import pathlib
import openset_imagenet
import os, sys
import torch
import numpy
from loguru import logger

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from collections import defaultdict

from matplotlib import pyplot, cm, colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, LogLocator

#def train_one(cmd):
#  print(cmd)
#  print(" ".join(cmd))

def command_line_options(command_line_arguments=None):
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Plotting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--protocols", "-p",
        type=int,
        choices = (1,2,3),
        nargs="+",
        default = (1,2,3),
        help="Select the protocols that should be evaluated"
    )
    parser.add_argument(
        "--losses", "-l",
        nargs = "+",
        choices = ('softmax', 'garbage', 'entropic'),
        default = ('softmax', 'garbage', 'entropic'),
        help = "Select the loss functions that should be included into the plot"
    )
    parser.add_argument(
        "--algorithms", "-a",
        choices = ["threshold", "maxlogits", "openmax", "evm", "proser"],
        nargs = "+",
        default = ["threshold", "maxlogits", "openmax", "evm", "proser"],
        help = "Which algorithm to include into the plot. Specific parameters should be in the yaml file"
    )
    parser.add_argument(
        "--configuration", "-c",
        type = pathlib.Path,
        default = pathlib.Path("config/test.yaml"),
        help = "The configuration file that defines some basic information"
    )
    parser.add_argument(
        "--use-best",
        action = "store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
    )
    parser.add_argument(
        "--force", "-f",
        action = "store_true",
        help = "If set, score files will be recomputed even if they already exist"
    )
    parser.add_argument(
        "--plots",
        help = "Select where to write the plots into"
    )
    parser.add_argument(
        "--tables",
        default = "results/Results_Protocol{}_{}.tex",
        help = "Select the files where to write the CCR table into"
    )
    parser.add_argument(
        "--fpr-thresholds", "-t",
        type = float,
        nargs="+",
        default = [1e-3, 1e-2, 1e-1, 1.],
        help = "Select the thresholds for which the CCR validation metric should be tabled"
    )

    args = parser.parse_args(command_line_arguments)

    args.plots = args.plots or f"results/Results_{'best' if args.use_best else 'last'}.pdf"
#    args.table = args.table or f"results/Results_{suffix}.tex"
    return args


def load_scores(args, cfg):
    # collect all result files;
    suffix = "best" if args.use_best else "curr"
    # we sort them as follows: protocol, loss, algorithm
    scores = defaultdict(lambda: defaultdict(dict))
    ground_truths = {}
#    epoch = {p:{} for p in args.protocols}
    for protocol in args.protocols:
        for loss in args.losses:
            for algorithm in args.algorithms:
                output_directory = pathlib.Path(cfg.output_directory) / f"Protocol_{protocol}"
                alg = "threshold" if algorithm == "maxlogits" else algorithm
                scr = "logits" if algorithm == "maxlogits" else "scores"
                score_file = output_directory / f"{loss}_{alg}_test_arr_{suffix}.npz"

                if os.path.exists(score_file):
                    # remember files
                    results = numpy.load(score_file)

                    scores[protocol][loss][algorithm] = results[scr]

                    if protocol not in ground_truths:
                        ground_truths[protocol] = results["gt"].astype(int)
                    else:
                        assert numpy.all(results["gt"] == ground_truths[protocol])

                    logger.info(f"Loaded score file {score_file} for protocol {protocol}, {loss}, {algorithm}")
                else:
                    logger.warning(f"Did not find score file {score_file} for protocol {protocol}, {loss}, {algorithm}")

    return scores, ground_truths


def plot_OSCR(args, scores, ground_truths):
    # plot OSCR
    P = len(args.protocols)
    fig = pyplot.figure(figsize=(8,3*P))
    gs = fig.add_gridspec(P, 2, hspace=0.25, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    axs = axs.flat
    font = 15

    for index, protocol in enumerate(args.protocols):
        openset_imagenet.util.plot_oscr(arrays=scores[protocol], gt=ground_truths[protocol], scale="semilog", title=f'$P_{protocol}$ Negative',
                    ax_label_font=font, ax=axs[2*index], unk_label=-1,)
        openset_imagenet.util.plot_oscr(arrays=scores[protocol], gt=ground_truths[protocol], scale="semilog", title=f'$P_{protocol}$ Unknown',
                    ax_label_font=font, ax=axs[2*index+1], unk_label=-2,)
    # Axis properties
    for ax in axs:
        ax.label_outer()
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

    # Figure labels
    fig.text(0.5, 0.06, 'FPR', ha='center', fontsize=font)
    fig.text(0.04, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)

    # add legend
    openset_imagenet.util.oscr_legend(
        args.losses, args.algorithms, fig,
        bbox_to_anchor=(0.5,-0.03), handletextpad=0.6, columnspacing=1.5,
        title="How to Read: Line Style -> Loss; Color -> Algorithm"
    )


def plot_OSCR_separated(args, scores, ground_truths, unk_label):
    # plot OSCR
    P = len(args.protocols)
    L = len(args.losses)
    fig = pyplot.figure(figsize=(4*L,2*P))
    gs = fig.add_gridspec(P, L, hspace=0.25, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    font = 15

    for p, protocol in enumerate(args.protocols):
        for l, loss in enumerate(args.losses):
            openset_imagenet.util.plot_oscr(arrays={loss: scores[protocol][loss]}, gt=ground_truths[protocol], scale="semilog", title=f'$P_{protocol}$ {NAMES[loss]}',
                    ax_label_font=font, ax=axs[p,l], unk_label=unk_label, line_style="solid")
    # Axis properties
    for ax in axs.flat:
        ax.label_outer()
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

    # Figure labels
    fig.text(0.5, 0.03, 'FPR', ha='center', fontsize=font)
    fig.text(0.07, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)

    # add legend
    openset_imagenet.util.oscr_legend(
        [args.losses[0]], args.algorithms, fig,
        bbox_to_anchor=(0.5,-0.04), handletextpad=0.6, columnspacing=1.5,
    )



from openset_imagenet.util import NAMES
def plot_score_distributions(args, scores, ground_truths, pdf):

    font_size = 15
    bins = 30
    P = len(args.protocols)
    L = len(args.losses)
    algorithms = [a for a in args.algorithms if a != "maxlogits"]
#    algorithms = args.algorithms
    A = len(algorithms)

    # Manual colors
    edge_unknown = colors.to_rgba('indianred', 1)
    fill_unknown = colors.to_rgba('firebrick', 0.04)
    edge_known = colors.to_rgba('tab:blue', 1)
    fill_known = colors.to_rgba('tab:blue', 0.04)
    edge_negative = colors.to_rgba('tab:green', 1)
    fill_negative = colors.to_rgba('tab:green', 0.04)

    for p, protocol in enumerate(args.protocols):

#        fig = pyplot.figure(figsize=(3*A+1, 2*P))
        fig = pyplot.figure(figsize=(3*A+1, 2*L))
        gs = fig.add_gridspec(L, A, hspace=0.2, wspace=0.08)
        axs = gs.subplots(sharex=True, sharey=False)

        for l, loss in enumerate(args.losses):
            max_a = (0, 0)

            for a, algorithm in enumerate(algorithms):
                # Calculate histogram
                ax = axs[l,a]
                if scores[protocol][loss][algorithm] is not None:
                    histograms = openset_imagenet.util.get_histogram(
                        scores[protocol][loss][algorithm],
                        ground_truths[protocol],
                        bins=bins
                    )
                    # Plot histograms
                    ax.stairs(histograms["known"][0], histograms["known"][1], fill=True, color=fill_known, edgecolor=edge_known, linewidth=1)
                    ax.stairs(histograms["unknown"][0], histograms["unknown"][1], fill=True, color=fill_unknown, edgecolor=edge_unknown, linewidth=1)
                    ax.stairs(histograms["negative"][0], histograms["negative"][1], fill=True, color=fill_negative, edgecolor=edge_negative, linewidth=1)

#                ax.set_title(f"{NAMES[protocol]} {NAMES[algorithm]}")
                ax.set_title(f"{NAMES[loss]} + {NAMES[algorithm]}")

                # set tick locator
                max_a = max(max_a, (max(numpy.max(h[0]) for h in histograms.values()), a))
                ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.label_outer()

            # share axis of the maximum value
            for a in range(A):
                if a != max_a[1]:
#                    axs[p,a].set_ylim([0, max_a[0]])
#                    axs[p,a].sharey(axs[p,max_a[1]])
                    axs[l,a].sharey(axs[l,max_a[1]])


        # Manual legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([None], [None], color=edge_known),
                           Line2D([None], [None], color=edge_negative),
                           Line2D([None], [None], color=edge_unknown)]
        legend_labels = ["Known", "Negative", "Unknown"]
        fig.legend(handles=legend_elements, labels=legend_labels, loc="lower right", ncol=3, bbox_to_anchor=(0.85,0.01))
#        fig.legend(handles=legend_elements, labels=legend_labels, loc="lower right", ncol=3)
        fig.text(0.3, 0.03, "Probability Score", ha='center', fontsize=font_size)
        fig.text(0.06, 0.5, "Sample Count", va='center', rotation='vertical', fontsize=font_size)


        # X label
#        fig.text(0.7, 0.01, f'{NAMES[loss]} Scores', ha='center', fontsize=font_size)

        pdf.savefig(bbox_inches='tight', pad_inches = 0)


from .parameter_selection import THRESHOLDS
def ccr_table(args, scores, gt):

    def nonemax(a,b):
        b = numpy.array([v if v is not None else numpy.nan for v in b])
        return numpy.where(numpy.logical_and(numpy.logical_not(numpy.isnan(b)), b >= a), b, a)
    for protocol in args.protocols:
        latex_file = args.tables.format(protocol, 'best' if args.use_best else 'last')
        print("Writing CCR tables for protocol", protocol, "to file", latex_file)
        # compute all CCR values and store maximum values
        results_n = collections.defaultdict(dict)
        max_total_n = numpy.zeros(len(args.fpr_thresholds)+1)
        max_by_alg_n = collections.defaultdict(lambda:numpy.zeros(len(args.fpr_thresholds)+1))
        max_by_loss_n = collections.defaultdict(lambda:numpy.zeros(len(args.fpr_thresholds)+1))
        results_u = collections.defaultdict(dict)
        max_total_u = numpy.zeros(len(args.fpr_thresholds)+1)
        max_by_alg_u = collections.defaultdict(lambda:numpy.zeros(len(args.fpr_thresholds)+1))
        max_by_loss_u = collections.defaultdict(lambda:numpy.zeros(len(args.fpr_thresholds)+1))
        for algorithm in args.algorithms:
            for loss in args.losses:
                auc = openset_imagenet.metrics.auc_score_binary(gt[protocol], scores[protocol][loss][algorithm], unk_label=-1)
                ccrs = openset_imagenet.util.ccr_at_fpr(gt[protocol], scores[protocol][loss][algorithm], args.fpr_thresholds, unk_label=-1)
                results_n[algorithm][loss] = [auc] + ccrs
                max_total_n = nonemax(max_total_n, results_n[algorithm][loss])
                max_by_alg_n[algorithm] = nonemax(max_by_alg_n[algorithm], results_n[algorithm][loss])
                max_by_loss_n[loss] = nonemax(max_by_loss_n[loss], results_n[algorithm][loss])

                auc = openset_imagenet.metrics.auc_score_binary(gt[protocol], scores[protocol][loss][algorithm], unk_label=-2)
                ccrs = openset_imagenet.util.ccr_at_fpr(gt[protocol], scores[protocol][loss][algorithm], args.fpr_thresholds, unk_label=-2)
                results_u[algorithm][loss] = [auc] + ccrs
                max_total_u = nonemax(max_total_u, results_u[algorithm][loss])
                max_by_alg_u[algorithm] = nonemax(max_by_alg_u[algorithm], results_u[algorithm][loss])
                max_by_loss_u[loss] = nonemax(max_by_loss_u[loss], results_u[algorithm][loss])


        with open(latex_file, "w") as f:
            # write header
            f.write("\\multirow{2}{*}{\\bf Algorithm} & \\multirow{2}{*}{\\bf Loss} & \\multicolumn{4}{c||}{\\bf Negative} & \\multicolumn{4}{c||}{\\bf Unknown} & \\bf Acc \\\\\\cline{3-11}\n & & ")
            f.write(" & ".join((["\\bf AUROC"] + [THRESHOLDS[t] for t in args.fpr_thresholds[:-1]]) * 2 + [THRESHOLDS[1]]))
            f.write("\\\\\\hline\\hline\n")
            for algorithm in args.algorithms:
                f.write(f"\\multirow{{{len(args.losses)}}}{{*}}{{{NAMES[algorithm]}}}")
                for loss in args.losses:
                    f.write(f" & {NAMES[loss]}")
                    for i, v in enumerate(results_n[algorithm][loss][:-1]):
                        if v is None: f.write(" &")
                        elif v == max_total_n[i]: f.write(f" & \\textcolor{{blue}}{{\\bf {v:.4f}}}")
                        elif v == max_by_alg_n[algorithm][i]: f.write(f" & \\it {v:.4f}")
                        elif v == max_by_loss_n[loss][i]: f.write(f" & \\underline{{{v:.4f}}}")
                        else: f.write(f" & {v:.4f}")
                    for i, v in enumerate(results_u[algorithm][loss]):
                        if v is None: f.write(" &")
                        elif v == max_total_u[i]: f.write(f" & \\textcolor{{blue}}{{\\bf {v:.4f}}}")
                        elif v == max_by_alg_u[algorithm][i]: f.write(f" & \\it {v:.4f}")
                        elif v == max_by_loss_u[loss][i]: f.write(f" & \\underline{{{v:.4f}}}")
                        else: f.write(f" & {v:.4f}")
                    f.write("\\\\\n")
                f.write("\\hline\n")


def main(command_line_arguments = None):
    args = command_line_options(command_line_arguments)
    cfg = openset_imagenet.util.load_yaml(args.configuration)

    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
#    logger.add(
#        sink = sys.stdout,
#        format=msg_format,
#        level="INFO")

    print("Extracting and loading scores")
    scores, ground_truths = load_scores(args, cfg)

    print("Writing file", args.plots)
    pdf = PdfPages(args.plots)
    try:
        # plot OSCR (actually not required for best case)
        print("Plotting OSCR curves")
        plot_OSCR(args, scores, ground_truths)
        pdf.savefig(bbox_inches='tight', pad_inches = 0)

        plot_OSCR_separated(args, scores, ground_truths, -1)
        pdf.savefig(bbox_inches='tight', pad_inches = 0)
        plot_OSCR_separated(args, scores, ground_truths, -2)
        pdf.savefig(bbox_inches='tight', pad_inches = 0)

        """
        if not args.linear and not args.use_best and not args.sort_by_loss:
          # plot confidences
          print("Plotting confidence plots")
          plot_confidences(args)
          pdf.savefig(bbox_inches='tight', pad_inches = 0)
        """

        # plot histograms
        print("Plotting score distribution histograms")
        plot_score_distributions(args, scores, ground_truths, pdf)
    finally:
        pdf.close()

    # create result table
    print("Creating CCR Tables")
    ccr_table(args, scores, ground_truths)
