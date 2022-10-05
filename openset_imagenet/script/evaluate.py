""" Independent code for inference in testing dataset. The functions are included and executed
in the train.py script."""
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from vast.tools import set_device_gpu, set_device_cpu, device
from torchvision import transforms as tf
from torch.utils.data import DataLoader

import openset_imagenet


def get_args():
    """Gets the evaluation parameters."""
    parser = argparse.ArgumentParser("Get parameters for evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # directory parameters
    parser.add_argument(
        "loss",
        choices = ["entropic", "softmax", "garbage"],
        help="Which loss function to evaluate"
    )
    parser.add_argument(
        "protocol",
        type = int,
        choices = (1,2,3),
        help = "Which protocol to evaluate"
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
        "--imagenet-directory",
        type=Path,
        default=Path("/local/scratch/datasets/ImageNet/ILSVRC2012/"),
        help="Imagenet root directory"
    )
    parser.add_argument(
        "--protocol-directory",
        type=Path,
        default = "protocols",
        help = "Where are the protocol files stored"
    )
    parser.add_argument(
        "--output-directory",
        default = "experiments/Protocol_{}",
        help = "Where to find the results of the experiments"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Select the batch size for the test set batches")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Data loaders number of workers, default:4")

    args = parser.parse_args()
    try:
        args.output_directory = args.output_directory.format(args.protocol)
    except:
        pass
    args.output_directory = Path(args.output_directory)
    return args



def main():
    args = get_args()

    # Create transformations
    transform_val = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    # create datasets
    val_dataset = openset_imagenet.ImagenetDataset(
        csv_file=args.protocol_directory/f"p{args.protocol}_val.csv",
        imagenet_path=args.imagenet_directory,
        transform=transform_val)

    test_dataset = openset_imagenet.ImagenetDataset(
        csv_file=args.protocol_directory/f"p{args.protocol}_test.csv",
        imagenet_path=args.imagenet_directory,
        transform=transform_val)

    # Info on console
    print("\n========== Data ==========")
    print(f"Val dataset len:{len(val_dataset)}, labels:{val_dataset.label_count}")
    print(f"Test dataset len:{len(test_dataset)}, labels:{test_dataset.label_count}")

    # create data loaders
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    # create device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
    else:
        print("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    if args.loss == "garbage":
        n_classes = val_dataset.label_count # we use one class for the negatives
    else:
        n_classes = val_dataset.label_count - 1  # number of classes - 1 when training with unknowns

    # create model
    suffix = "_best" if args.use_best else "_curr"
    model = openset_imagenet.ResNet50(fc_layer_dim=n_classes, out_features=n_classes, logit_bias=False)
    start_epoch, best_score = openset_imagenet.train.load_checkpoint(model, args.output_directory / (args.loss+suffix+".pth"))
    print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)

    print("========== Evaluating ==========")
    print("Validation data:")
    # extracting arrays for validation
    gt, logits, features, scores = openset_imagenet.train.get_arrays(
        model=model,
        loader=val_loader
    )
    file_path = args.output_directory / f"{args.loss}_val_arr{suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")

    # extracting arrays for test
    print("Test data:")
    gt, logits, features, scores = openset_imagenet.train.get_arrays(
        model=model,
        loader=test_loader
    )
    file_path = args.output_directory / f"{args.loss}_test_arr{suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")
