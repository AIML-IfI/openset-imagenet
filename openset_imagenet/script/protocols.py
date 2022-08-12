"""Open set protocols V2"""
from pathlib import Path
import argparse

from openset_imagenet import OpenSetProtocol

def get_args():
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Protocols Parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--prot",
        type=int,
        nargs="+",
        default=(1,2,3),
        help="Open set protocol: 1, 2 or 3")
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default="/local/scratch/datasets/ImageNet/ILSVRC2012/",
        help="Path to imagenet ILSVRC2012 dataset, it must contain train and val folders")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default="/local/scratch/datasets/ImageNet/ILSVRC2012/robustness",
        help="Directory of metadata files (imagenet_class_index.json, wordnet.is_a.txt, words.txt)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default="protocols",
        help="Directory to save protocol files")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Integer random seed; handle with care: different seeds will provide different protocols!")

    parser = parser.parse_args()
    return parser

def main():
    args = get_args()
    for prot in args.prot:
        protocol = OpenSetProtocol(
            imagenet_dir=args.imagenet_dir,
            metadata_path=args.metadata_dir,
            protocol_num=prot)
        protocol.create_dataset(random_state=args.seed)
        protocol.print_data()
        protocol.save_datasets_to_csv(args.out_dir)
