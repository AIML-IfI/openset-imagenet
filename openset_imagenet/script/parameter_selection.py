""" Independent code for parameter selection in the validation dataset.
For EVM and OpenMax, this requires that the according training has been performed already.
For PROSER, training is run by this script."""

import argparse
import pathlib
import numpy
import sys, os
import collections
from vast.tools import set_device_gpu, set_device_cpu, device
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import openset_imagenet
import pickle
from openset_imagenet.openmax_evm import compute_adjust_probs, compute_probs, get_param_string
from openset_imagenet.util import NAMES
from loguru import logger
import tqdm

def command_line_options(command_line_arguments=None):
    """Gets the evaluation parameters."""
    parser = argparse.ArgumentParser("Get parameters for evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # directory parameters
    parser.add_argument(
        "--losses", "-l",
        choices = ["softmax", "garbage", "entropic"],
        nargs="+",
        default = ["softmax", "garbage", "entropic"],
        help="Which loss functions to optimize on"
    )
    parser.add_argument(
        "--protocols", "-p",
        type = int,
        nargs = "+",
        choices = (1,2,3),
        default = (1,2,3),
        help = "Which protocols to optimize on"
    )
    parser.add_argument(
        "--algorithms", "-a",
        choices = ["openmax", "evm"],
        nargs = "+",
        default = ["openmax", "evm"],
        help = "Which algorithms to optimize. Specific parameters should be in the yaml files"
    )
    parser.add_argument(
        "--configuration-directory",
        type = pathlib.Path,
        default = pathlib.Path("config"),
        help = "The directory containing all base configuration files"
    )
    parser.add_argument(
        "--fpr-thresholds", "-t",
        type = float,
        nargs="+",
        default = [1e-3, 1e-2, 1e-1, 1.],
        help = "Select the thresholds for which the CCR validation metric should be computed"
    )
    parser.add_argument(
        "--use-best", "-b",
        action="store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
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
        "--latex-files",
        default = "results/Parameters_{}_{}_{}.tex",
        help = "Set the output file where to write the parameter tables into; the value will get formatted with protocol, loss and algorithm"
    )
    parser.add_argument(
        "--summary-file",
        default = "results/Parameter_summary_{}.tex",
        help = "Select the file where to write the final summary table into"
    )

    args = parser.parse_args(command_line_arguments)
    return args


def dataset(cfg, protocol):
    # Create transformations
    transform = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    # We only need validation data here
    val_dataset = openset_imagenet.ImagenetDataset(
        csv_file=cfg.data.val_file.format(protocol),
        imagenet_path=cfg.data.imagenet_path,
        transform=transform)

    val_dataset.re_order_labels()

    # Info on console
    logger.info(f"Loaded validation dataset for protocol {protocol} with len:{len(val_dataset)}, labels:{val_dataset.label_count}")

    # create data loaders
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    # return test loader
    return val_dataset, val_loader

from .evaluate_algs import load_model, extract


def post_process(gt, logits, features, scores, cfg, thresholds, protocol, loss, algorithm, output_directory, gpu):
    opt = cfg.algorithm

    parameters = {}

    if algorithm == "openmax":
        for tailsize in opt.tailsize:
            for distance_multiplier in opt.distance_multiplier:
                key = get_param_string("openmax", tailsize=tailsize, distance_multiplier=distance_multiplier)
                parameters[key] = (tailsize, distance_multiplier)
    elif algorithm == "evm":
        for tailsize in opt.tailsize:
            for distance_multiplier in opt.distance_multiplier:
                for cover_threshold in opt.cover_threshold:
                    key = get_param_string("evm", tailsize=tailsize, distance_multiplier=distance_multiplier, cover_threshold=cover_threshold)
                    parameters[key] = (tailsize, distance_multiplier, cover_threshold)

    results = {}

    # iterate through all keys and process our data
    logger.info(f"Evaluating {len(parameters)} sets of {algorithm} parameters for protocol {protocol}, {loss}")
    hyperparams = openset_imagenet.util.NameSpace(dict(distance_metric = opt.distance_metric))
    for key, params in tqdm.tqdm(parameters.items()):
        logger.debug(f"Evaluating {params}")
        model_path = opt.output_model_path.format(output_directory, loss, algorithm, key, opt.distance_metric)
        if not os.path.exists(model_path):
            logger.warning(f"Could not load model file {model_path}; skipping")
            continue
        model_dict = pickle.load(open(model_path, "rb"))

        if algorithm == 'openmax':
            for alpha in opt.alpha:
                new_scores = compute_adjust_probs(gt, logits, features, scores, model_dict, "openmax", gpu, hyperparams, alpha)
                results[params + (alpha,)] = openset_imagenet.util.ccr_at_fpr(gt, numpy.array(new_scores), thresholds, unk_label=-1)
        elif algorithm == 'evm':
            new_scores = compute_probs(gt, logits, features, scores, model_dict, "evm", gpu, hyperparams)
            results[params] = openset_imagenet.util.ccr_at_fpr(gt, numpy.array(new_scores), thresholds, unk_label=-1)

    return results


def process_model(protocol, loss, algorithm, cfg, thresholds, suffix, gpu):
    output_directory =pathlib.Path(cfg.output_directory)/f"Protocol_{protocol}"
    # set device
    if gpu is not None:
        set_device_gpu(index=gpu)
    else:
        logger.warning("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    # get dataset
    val_dataset, val_loader = dataset(cfg, protocol)

    # load base model
    if loss == "garbage":
        n_classes = val_dataset.label_count # we use one class for the negatives; the dataset has one additional label: negative
    else:
        n_classes = val_dataset.label_count - 1  # number of classes - 1 when training was without garbage class

    base_model = load_model(cfg, loss, "threshold", protocol, suffix, output_directory, n_classes)
    if base_model is None:
        logger.warning(f"The base model for protocol {protocol} and {loss} could not be found -- skipping")
        return

    # extract features
    logger.info(f"Extracting base scores for protocol {protocol}, {loss}")
    gt, logits, features, scores = extract(base_model, val_loader, "threshold", loss)
    # remove model from GPU memory
    del base_model

    # get results
    return post_process(gt, logits, features, scores, cfg, thresholds, protocol, loss, algorithm, output_directory, gpu)


HEADERS = {
    "evm": "$\\lambda$ & $\\kappa$ & $\\omega$ & ",
    "openmax": "$\\lambda$ & $\\kappa$ & $\\alpha$ & ",
}
THRESHOLDS = {1e-4: "$10^{-4}$",
              1e-3: "$10^{-3}$",
              1e-2: "$10^{-2}$",
              1e-1: "$10^{-1}$",
              1: "$1$",
}

def result_table(results, thresholds, latex_file, algorithm, write_header=False):

    # append the sum to the results
    max_values = [0] * (len(thresholds)+1)
    max_param = None
    for param, values in results.items():
        total = sum(v for v in values if v is not None)
        values.append(total)
        max_param = max_param if max_values[-1] > total else param
        for i, v in enumerate(values):
            if v is not None and v > max_values[i]:
                max_values[i] = v

    with open(latex_file, "w") as f:
        # write header
        if write_header:
            f.write(HEADERS[algorithm])
        # write FPR thresholds
        f.write(" & ".join([THRESHOLDS[t] for t in thresholds]))
        f.write(" & $\\Sigma$ \\\\\\hline\\hline\n")

        # write data rows
        for keys, values in results.items():
            f.write(" & ".join(
                ([f"{k}" for k in keys] if write_header else []) +
                [("\\bf " if v == max_values[i] else "") + f"{v:.4f}" if v is not None else "" for i,v in enumerate(values)]
            ))
            f.write(" \\\\\\hline\n")
    return max_param, max_values[-1]


def summary_table(maxima, summary_file):
    for algorithm in maxima:
        filepath = summary_file.format(algorithm)
        with open(filepath, "w") as f:
            # write header
            f.write("\\bf Protocol & \\bf Loss & ")
            f.write(HEADERS[algorithm])
            f.write("$\\Sigma$ \\\\\\hline\\hline\n")
            for protocol in maxima[algorithm]:
                # write protocol
                f.write(f"\\multirow{{{len(maxima[algorithm][protocol])}}}{{*}}{{{NAMES[protocol]}}}")
                for loss, values in maxima[algorithm][protocol].items():
                    f.write(f" & {NAMES[loss]} & ")
                    keys, value = values
                    f.write(" & ".join([str(k) for k in keys]))
                    f.write(f" & {value:.4f} \\\\\n")
                f.write("\\hline\n")
        logger.info(f"Wrote Summary file {filepath}")


def main(command_line_arguments=None):

    args = command_line_options(command_line_arguments)
    suffix = "best" if args.use_best else "curr"

    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])

    maxima = collections.defaultdict(lambda:collections.defaultdict(dict))

    for algorithm in args.algorithms:
      # load configuration for this algorithm
      cfg = openset_imagenet.util.load_yaml(pathlib.Path(args.configuration_directory) / (algorithm + ".yaml"))
      for loss in args.losses:
          for protocol in args.protocols:
              result = process_model(protocol, loss, algorithm, cfg, args.fpr_thresholds, suffix, args.gpu)
              if result is not None:
                  latex_file = args.latex_files.format(protocol, loss, algorithm)
                  maximum = result_table(result, args.fpr_thresholds, latex_file, algorithm, write_header=loss==args.losses[0])
                  logger.info(f"Wrote table '{latex_file}'")
                  maxima[algorithm][protocol][loss] = maximum

    summary_table(maxima, args.summary_file)
    # Write best results to console
    print("BEST RESULTS:")
    for algorithm in args.algorithms:
      for protocol in args.protocols:
        for loss in args.losses:
          print(f"{algorithm} in protocol {protocol} for {loss}: ", end="")
          h = HEADERS[algorithm].split(" & ")[:-1]
          m,v = maxima[algorithm][protocol][loss]
          print(", ".join([f"{h[i]}: {m[i]}" for i in range(len(h))]), end="")
          print(f", value: {v}")
        print()
      print()
