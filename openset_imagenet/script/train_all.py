"""Training of all models for the paper"""

import argparse
import multiprocessing
import subprocess
import pathlib
import openset_imagenet
import os

#def train_one(cmd):
#  print(cmd)
#  print(" ".join(cmd))

def command_line_options(command_line_arguments=None):
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Training Parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--configuration-directory",
        type = pathlib.Path,
        default = pathlib.Path("config"),
        help = "The directory containing all base configuration files"
    )
    parser.add_argument(
        "--protocols", "-p",
        type=int,
        choices = (1,2,3),
        nargs="+",
        default = (3,1,2),
        help="Select the protocols that should be executed"
    )
    parser.add_argument(
      "--loss-functions", "-l",
      nargs = "+",
      choices = ('entropic', 'softmax', 'garbage'),
      default = ('entropic', 'softmax', 'garbage'),
      help = "Select the loss functions that should be evaluated"
      )

    parser.add_argument(
      "--algorithms", "-a",
      nargs = "+",
      choices = ('threshold', 'evm', 'openmax', 'proser'),
      default = ('threshold', 'evm', 'openmax', 'proser')
    )
    parser.add_argument(
        "--output-directory", "-o",
        type=pathlib.Path,
        default="experiments",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--gpus", "-g",
        type = int,
        nargs="+",
        help = "Select the GPU indexes that you want to use. If you specify more than one index, training will be executed in parallel."
    )
    parser.add_argument(
        "--nice",
        type=int,
        default = 20,
        help = "Select priority level"
    )
    parser.add_argument(
        "--continue-training", "-c",
        action = "store_true",
        help = "Continue training when old snapshot is available"
    )

    args = parser.parse_args(command_line_arguments)
    args.parallel = args.gpus is not None and len(args.gpus) > 1
    return args

def commands(args):
  gpu = 0
  gpus = len(args.gpus) if args.gpus is not None else 1
  processes = [[] for _ in range(gpus)]
  for protocol in args.protocols:
    for loss_function in args.loss_functions:
      # make sure that the base model has been trained
      for algorithm in args.algorithms:
        # load base config file
        config = openset_imagenet.util.load_yaml(pathlib.Path(args.configuration_directory) / (algorithm + ".yaml"))
        outdir = os.path.join(args.output_directory, f"Protocol_{protocol}")
        # modify config file
        config.loss.type = loss_function
        config.name = loss_function
        config.parallel = args.parallel
        config.log_name = loss_function + "_" + algorithm + ".log"
        config.output_directory = outdir
        config.protocol = protocol
        # check to continue
        if args.continue_training:
          checkpoint_file = config.algorithm.output_model_path.format(config.output_directory, config.loss.type, config.algorithm.type, config.epochs, config.algorithm.dummy_counts[0], "curr") if algorithm == "proser" else config.model_path.format(outdir, loss_function, algorithm, "curr")
          if os.path.exists(checkpoint_file):
            config.checkpoint = checkpoint_file

        # write config file
        config_file = os.path.join(outdir, loss_function + "_" + algorithm + ".yaml")
        os.makedirs(outdir, exist_ok=True)
        open(config_file, "w").write(config.dump())

        call = ["train_imagenet.py", config_file, str(protocol), "--nice", str(args.nice)]
        if args.gpus is not None:
          call += ["--gpu", str(args.gpus[gpu])]
          processes[gpu].append(call)
          gpu = (gpu + 1) % gpus
        else:
          processes[0].append(call)

  return processes

def train_one_gpu(processes):
  for process in processes:
    print("Running experiment: " + " ".join(process))
    subprocess.call(process)

def main():
  args = command_line_options()
  if args.parallel:
    # we run in parallel
    with multiprocessing.pool.ThreadPool(len(args.gpus)) as pool:
      pool.map(train_one_gpu, commands(args))
  else:
    for c in commands(args):
      train_one_gpu(c)
