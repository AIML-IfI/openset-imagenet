"""Training of all models for the paper"""

import argparse
import multiprocessing
import collections
import subprocess
import pathlib
import openset_imagenet
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from matplotlib import pyplot, cm, colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, LogLocator

#def train_one(cmd):
#  print(cmd)
#  print(" ".join(cmd))

def get_args():
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Plotting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--protocols",
        type=int,
        choices = (1,2,3),
        nargs="+",
        default = (1,2,3),
        help="Select the protocols that should be evaluated"
    )
    parser.add_argument(
        "--loss-functions", "-l",
        nargs = "+",
        choices = ('softmax', 'entropic', 'garbage'),
        default = ('softmax', 'entropic', 'garbage'),
        help = "Select the loss functions that should be evaluated"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices = ("S", "EOS", "BG"),
        default = ("S", "EOS", "BG"),
        help = "Select the labels for the plots"
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
        "--output-directory", "-o",
        type=pathlib.Path,
        default="experiments",
        help="Directory where the models are saved"
    )
    parser.add_argument(
        "--imagenet-directory",
        type=pathlib.Path,
        default=pathlib.Path("/local/scratch/datasets/ImageNet/ILSVRC2012/"),
        help="Imagenet root directory"
    )
    parser.add_argument(
        "--protocol-directory",
        type=pathlib.Path,
        default = "protocols",
        help = "Where are the protocol files stored"
    )
    parser.add_argument(
        "--gpu", "-g",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
    )
    parser.add_argument(
      "--plots",
      default = "ImageNet.pdf",
      help = "Select where to write the plots into"
    )

    args = parser.parse_args()
    return args


def plot_OSCR(args):

    # collect all result files
    val = {p:{} for p in args.protocols}
    test = {p:{} for p in args.protocols}
    for protocol in args.protocols:
      for loss in args.loss_functions:
        experiment_dir = args.output_directory / f"Protocol_{protocol}"
        suffix = "_best" if args.use_best else "_curr"
        checkpoint_file = experiment_dir / (loss+suffix+".pth")
        score_files = {v : experiment_dir / f"{loss}_{v}_arr{suffix}.npz" for v in ("val", "test")}
        if os.path.exists(checkpoint_file):
          if not all(os.path.exists(v) for v in score_files.values()) or args.force:
            # extract score files first
            print("Extracting scores of", checkpoint_file)
            call = ["evaluate_imagenet.py", loss, str(protocol), "--output-directory", experiment_dir, "--imagenet-directory", args.imagenet_directory, "--protocol-directory", args.protocol_directory]
            if args.gpu is not None:
              call += ["-g", str(args.gpu)]
            if args.use_best:
              call += ["-b"]
            subprocess.call(call)
          # remember files
          scores = openset_imagenet.util.read_array_list(score_files)
          val[protocol][loss] = scores["val"]
          test[protocol][loss] = scores["test"]
        else:
          print ("Checkpoint file", checkpoint_file, "not found, skipping protocol", protocol, loss)
          val[protocol][loss] = None
          test[protocol][loss] = None

    # plot OSCR
    P = len(args.protocols)
    fig = pyplot.figure(figsize=(5*P,6))
    gs = fig.add_gridspec(2, P, hspace=0.2, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    axs = axs.flat
    font = 15
    scale = 'semilog'

    for index, p in enumerate(args.protocols):
      openset_imagenet.util.plot_oscr(arrays=val[p], scale=scale, title=f'$P_{p}$ Validation',
                    ax_label_font=font, ax=axs[index], unk_label=-1,)
      openset_imagenet.util.plot_oscr(arrays=test[p], scale=scale, title=f'$P_{p}$ Test',
                    ax_label_font=font, ax=axs[index+P], unk_label=-2,)
    # Manual legend
    axs[-P].legend(args.labels, frameon=False,
              fontsize=font - 1, bbox_to_anchor=(0.8, -0.12), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
    # Axis properties
    for ax in axs:
        ax.label_outer()
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')
    # Figure labels
    fig.text(0.5, 0.03, 'FPR', ha='center', fontsize=font)
    fig.text(0.08, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)
    return val, test


def plot_confidences(args):

  # locate event paths
  event_files = {p:collections.defaultdict(list) for p in args.protocols}
  for protocol in args.protocols:
    protocol_dir = args.output_directory/f"Protocol_{protocol}"
    if os.path.exists(protocol_dir):
      files = sorted(os.listdir(protocol_dir))
      # get the event files
      for f in files:
        if f.startswith("event"):
          loss = f.split("-")[1].split(".")[0]
          # set (overwrite) event file
          event_files[protocol][loss].append(protocol_dir / f)

  P = len(args.protocols)
  linewidth = 1.5
  legend_pos = "lower right"
  font_size = 15
  color_palette = cm.get_cmap('tab10', 10).colors
  fig = pyplot.figure(figsize=(12,3*P+1))
  gs = fig.add_gridspec(P, 2, hspace=0.2, wspace=0.1)
  axs = gs.subplots(sharex=True, sharey=True)
  axs = axs.flat


  def load_accumulators(files):
    known_data, unknown_data = {}, {}
    for event_file in files:
      try:
        event_acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
        event_acc.Reload()
        for event in event_acc.Scalars('val/conf_kn'):
          known_data[event[1]] = event[2]
        for event in event_acc.Scalars('val/conf_unk'):
          unknown_data[event[1]] = event[2]
      except KeyError: pass

    # re-structure
    return zip(*sorted(known_data.items())), zip(*sorted(unknown_data.items()))

  max_len = 0
  for index, protocol in enumerate(args.protocols):
      ax_kn = axs[2 * index]
      ax_unk = axs[2 * index + 1]
      for c, loss in enumerate(args.loss_functions):
        step_kn, val_kn, step_unk, val_unk = [[]]*4
        if loss in event_files[protocol]:
          # Read data from the tensorboard object
          (step_kn, val_kn), (step_unk, val_unk) = load_accumulators(event_files[protocol][loss])
        else:
          step_kn, val_kn, step_unk, val_unk = [[]]*4

        # Plot known confidence
        ax_kn.plot(step_kn, val_kn, linewidth=linewidth, label = loss + ' kn', color=color_palette[c])
        # Plot unknown confidence
        ax_unk.plot(step_unk, val_unk, linewidth=linewidth, label = loss + ' unk', color=color_palette[c])
        if len(step_kn):
          max_len = max(max_len, max(step_kn))
      # set titles
      ax_kn.set_title(f"$P_{protocol}$ Known", fontsize=font_size)
      ax_unk.set_title(f"$P_{protocol}$ Negative", fontsize=font_size)

  # Manual legend
  axs[-2].legend(args.labels, frameon=False,
                fontsize=font_size - 1, bbox_to_anchor=(0.8, -0.1), ncol=3, handletextpad=0.5, columnspacing=1)

  for ax in axs:
      # set the tick parameters for the current axis handler
      ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
      ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
      ax.set_xlim(0, max_len)
      ax.set_ylim(0, 1)
      # Thicklocator parameters
      ax.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
      ax.xaxis.set_major_locator(MaxNLocator(6))
      ax.label_outer()
  # X label
  fig.text(0.5, 0.05, 'Epoch', ha='center', fontsize=font_size)


def plot_softmax(args, test):

    font_size = 15
    bins = 30
    unk_label = -2
    P = len(args.protocols)
    N = len(args.loss_functions)

    fig = pyplot.figure(figsize=(3*P+1, 3*N))
    gs = fig.add_gridspec(N, P, hspace=0.2, wspace=0.08)
    axs = gs.subplots(sharex=True, sharey=False)
    axs = axs.flat
    # Manual colors
    edge_unk = colors.to_rgba('indianred', 1)
    fill_unk = colors.to_rgba('firebrick', 0.04)
    edge_kn = colors.to_rgba('tab:blue', 1)
    fill_kn = colors.to_rgba('tab:blue', 0.04)

    index = 0
    for protocol in args.protocols:
      for l, loss in enumerate(args.loss_functions):
        # Calculate histogram
        drop_bg = loss == "garbage"  #  Drop the background class
        if test[protocol][loss] is not None:
          kn_hist, kn_edges, unk_hist, unk_edges = openset_imagenet.util.get_histogram(
              test[protocol][loss],
              unk_label=unk_label,
              metric='score',
              bins=bins,
              drop_bg=drop_bg
          )
        else:
          kn_hist, kn_edges, unk_hist, unk_edges = [], [0], [], [0]
        # Plot histograms
        axs[index].stairs(kn_hist, kn_edges, fill=True, color=fill_kn, edgecolor=edge_kn, linewidth=1)
        axs[index].stairs(unk_hist, unk_edges, fill=True, color=fill_unk, edgecolor=edge_unk, linewidth=1)

        # axs[ix].set_yscale('log')
        axs[index].set_title(f"${args.labels[l]}_{{{protocol}}}$")
        index += 1

    # Share y axis of the histograms of the same protocol
    for p in range(P):
      for l in range(1,N):
        axs[N*p+l-1].sharey(axs[N*p+l])

    for ax in axs:
        # set the tick parameters for the current axis handler
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.label_outer()

    # Manual legend
    axs[-2].legend(['Known', 'Unknown'],
                  frameon=False,
                  fontsize=font_size-1,
                  bbox_to_anchor=(0.2, -0.08),
                  ncol=2,
                  handletextpad=0.3,
                  columnspacing=1,
                  markerscale=1)
    # X label
    fig.text(0.5, 0.06, 'Score', ha='center', fontsize=font_size)

def main():
  args = get_args()


  pdf = PdfPages(args.plots)
  try:
    # plot OSCR
    print("Plotting OSCR curves")
    val, test = plot_OSCR(args)
    pdf.savefig(bbox_inches='tight', pad_inches = 0)

    # plot confidences
    print("Plotting confidence plots")
    plot_confidences(args)
    pdf.savefig(bbox_inches='tight', pad_inches = 0)

    # plot histograms
    print("Plotting softmax histograms")
    plot_softmax(args, test)
    pdf.savefig(bbox_inches='tight', pad_inches = 0)


  finally:
    pdf.close()
