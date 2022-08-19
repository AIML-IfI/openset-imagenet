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
        "--protocols",
        type=int,
        nargs="+",
        default=(1,2,3),
        help="Open set protocol: 1, 2 or 3")
    parser.add_argument(
        "--imagenet-directory",
        type=Path,
        default="/local/scratch/datasets/ImageNet/ILSVRC2012/",
        help="Path to imagenet ILSVRC2012 dataset, it must contain train and val folders")
    parser.add_argument(
        "--metadata-directory",
        type=Path,
        default="/local/scratch/datasets/ImageNet/ILSVRC2012/robustness",
        help="Directory of metadata files (imagenet_class_index.json, wordnet.is_a.txt, words.txt)")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="protocols",
        help="Directory to save protocol files")
    parser.add_argument(
        "--tex-files", "-t",
        type = Path,
        nargs="+",
        help = "Write the list of classes into the provided files, one per protocol"
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Integer random seed; handle with care: different seeds will provide different protocols!")

    args = parser.parse_args()
    if args.tex_files is not None and len(args.tex_files) != len(args.protocols):
        raise ValueError(f"If specified, the number of --tex-files {len(args.tex_files)} and --protocols {len(args.protocols)} need to be identical")
    return args

def main():
    args = get_args()
    for i, prot in enumerate(args.protocols):
        protocol = OpenSetProtocol(
            imagenet_dir=args.imagenet_directory,
            metadata_path=args.metadata_directory,
            protocol_num=prot)
        protocol.create_dataset(random_state=args.seed)
        protocol.print_data()
        protocol.save_datasets_to_csv(args.output_directory)
        if args.tex_files:
            protocol.write_class_list(args.tex_files[i])
