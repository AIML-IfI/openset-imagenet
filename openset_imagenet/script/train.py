""" Training script for Open-set Classification on Imagenet"""
import argparse
import openset_imagenet
import pathlib
import os


def get_args(command_line_options = None):
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
        "protocol",
        type=int,
        choices = (1,2,3),
        help="Open set protocol: 1, 2 or 3"
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
        "--nice",
        type=int,
        default = 20,
        help = "Select Priority Level"
    )

    args = parser.parse_args(command_line_options)

    os.nice(args.nice)
    return args


def main(command_line_options = None):

    args = get_args(command_line_options)
    config = openset_imagenet.util.load_yaml(args.configuration)
    if args.gpu is not None:
        config.gpu = args.gpu
    config.protocol = args.protocol

    if config.algorithm.type == "threshold":
        openset_imagenet.train.worker(config)
    elif config.algorithm.type in ['openmax', 'evm']:
        openset_imagenet.openmax_evm.worker(config)
    elif config.algorithm.type == "proser":
        openset_imagenet.proser.worker(config, 0)
    else:
        raise ValueError(f"The training configuration type '{config.algorithm.type}' is not known to the system")


if __name__ == "__main__":
    main()
