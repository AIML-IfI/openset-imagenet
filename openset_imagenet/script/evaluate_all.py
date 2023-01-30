import argparse
import multiprocessing
from pathlib import Path
import openset_imagenet
import os
import torch
import numpy
import multiprocessing
import subprocess


def get_args():
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Training Parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--configuration",
        type = Path,
        default = Path("./config/parameters.yaml"),
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
      choices = ('entropic', 'softmax', 'garbage'),
      default = ('entropic', 'softmax', 'garbage'),
      help = "Select the loss functions that should be evaluated"
      )
    parser.add_argument(
        "--output-directory", "-o",
        type=Path,
        default="experiments/Protocol_{}",
        help="Directory to save trained models"
    )

    parser.add_argument(
        "--algorithm", "-alg",
        choices = ["threshold", "openmax", "proser", "evm", "maxlogit"],
        help = "Which algorithm to evaluate. Specific parameters should be in the yaml file"
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
    try:
        args.output_directory = args.output_directory.format(args.protocol)
    except:
        pass
    args.output_directory = Path(args.output_directory)

    args.parallel = args.gpus is not None and len(args.gpus) > 1
    return args




def train_one_gpu(processes):
  for process in processes:
    print("Running experiment: " + " ".join(process))
    subprocess.call(process)


def main():

    args = get_args()
    #args = get_args(command_line_options)
    cfg = openset_imagenet.util.load_yaml(args.configuration)

    print(cfg.algorithm.type)
    if args.gpus:
        cfg.gpu = args.gpus[0]

    cfg.protocol = args.protocols
    cfg.algorithm.type = args.algorithm
    cfg.output_directory = args.output_directory

    gpu = 0
    gpus = len(args.gpus) if args.gpus is not None else 1
    processes = [[] for _ in range(1)]

    if cfg.algorithm.type == 'openmax':
        for ts in cfg.algorithm.tailsize:
            for dm in cfg.algorithm.distance_multiplier:
                    for alpha in cfg.algorithm.alpha_om:
                        print(ts, dm, alpha)
                        #python openset_imagenet/script/evaluate_algs.py config/test.yaml softmax 1 --gpu 1 --use-best -alg openmax
                        #call = ["python openset_imagenet/script/evaluate_algs.py", "config/test.yaml", "softmax", "1", "--gpu 1", "-alg openmax"]
                        subprocess.run(["python", "openset_imagenet/script/evaluate_algs.py", "config/test.yaml", "softmax", "1", "--gpu", "1", "-alg", "openmax"])

        '''                if args.gpus is not None:
                            call += ["--gpu", str(args.gpus[gpu])]
                            processes[gpu].append(call)
                            gpu = (gpu + 1) % gpus
                        else:
                            processes[0].append(call)


    train_one_gpu(processes) '''

if __name__=="__main__":
    main()
