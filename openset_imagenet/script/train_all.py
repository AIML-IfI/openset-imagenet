"""Training of all models for the paper"""

import argparse
import multiprocessing
import subprocess
import pathlib
from .train import main as train_one
import openset_imagenet
import os

#def train_one(cmd):
#  print(cmd)
#  print(" ".join(cmd))

def get_args():
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Training Parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "configuration",
        type = pathlib.Path,
        help = "The configuration file that defines the experiment"
    )
    parser.add_argument(
        "--protocols",
        type=int,
        choices = (1,2,3),
        nargs="+",
        default = (3,1,2),
        help="Select the protocols that should be executed"
    )
    parser.add_argument(
      "--loss-functions", "-l",
      nargs = "+",
      choices = ('objectosphere', 'entropic', 'softmax', 'garbage'),
      default = ('objectosphere', 'entropic', 'softmax', 'garbage'),
      help = "Select the loss functions that should be evaluated"
      )
    parser.add_argument(
        "--output-directory", "-o",
        type=pathlib.Path,
        default="experiments",
        help="Directory to save protocol files"
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

    args = parser.parse_args()
    args.parallel = args.gpus is not None and len(args.gpus) > 1
    return args

def commands(args):
  gpu = 0
  for protocol in args.protocols:
    for loss_function in args.loss_functions:
      # load config file
      config = openset_imagenet.util.load_yaml(args.configuration)
      # modify config file
      config.loss.type = loss_function
      config.name = loss_function
      config.parallel = args.parallel
      # write config file
      outdir = os.path.join(args.output_directory, f"Protocol_{protocol}")
      config_file = os.path.join(outdir, loss_function + ".yaml")
      os.makedirs(outdir, exist_ok=True)
      open(config_file, "w").write(config.dump())
      gpu += 1

#      call = ["train_imagenet.py", config_file, str(protocol), "--output-directory", outdir, "--nice", str(args.nice)]
      call = [config_file, str(protocol), "--output-directory", outdir, "--nice", str(args.nice)]
      if args.gpus is not None:
        call += ["--gpu", str(args.gpus[gpu % len(args.gpus)])]

      print("Running experiment: " + " ".join(call))
      yield call

#def train_one(call):
#  subprocess.call(call)

def main():
  args = get_args()
  if args.parallel:
    # we run in parallel
    with multiprocessing.Pool(len(args.gpus), maxtasksperchild=1) as pool:
      pool.map(train_one, commands(args))
  else:
    for c in commands(args):
      train_one(c)
