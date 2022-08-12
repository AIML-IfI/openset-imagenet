""" Training script for Open-set Classification on Imagenet"""
import argparse
import openset_imagenet.train
import pathlib


def get_args():
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Training Parameters")
    parser.add_argument(
        "configuration",
        type = pathlib.Path,
        help = "The configuration file that defines the experiment"
    )
    parser.add_argument(
        "protocol",
        type=int,
        help="Open set protocol: 1, 2 or 3")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=".",
        help="Directory to save protocol files")
    parser.add_argument(
        "--gpu",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
        )

    args = parser.parse_args()
    return args


def main():

    args = get_args()
    config = openset_imagenet.util.load_yaml(args.configuration)
    config.

    openset_imagenet.train.worker(args.gpu, cfg, out_dir,)




if __name__ == "__main__":
    main()
