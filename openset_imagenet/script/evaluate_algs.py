""" Independent code for inference in testing dataset. The functions are included and executed
in the train.py script."""
import argparse
from pathlib import Path
import numpy as np
import torch
from vast.tools import set_device_gpu, set_device_cpu, device
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import openset_imagenet
import pickle
from openset_imagenet.openmax_evm import compute_adjust_probs, compute_probs, compose_dicts, get_ccr_at_fpr, validate, get_param_string

def command_line_options(command_line_arguments=None):
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
        "--configuration", "-c",
        type = Path,
        default = Path("config/test.yaml"),
        help = "The configuration file that defines the experiment"
    )

    parser.add_argument(
        "--algorithm", "-a",
        choices = ["threshold", "openmax", "proser", "evm"],
        help = "Which algorithm to evaluate. Specific parameters should be in the yaml file"
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

    args = parser.parse_args(command_line_arguments)
    return args


def main(command_line_arguments = None):

    args = command_line_options(command_line_arguments)
    cfg = openset_imagenet.util.load_yaml(args.configuration)
    cfg.protocol = args.protocol

    output_directory = Path(cfg.output_directory)/f"Protocol_{args.protocol}"

    # Create transformations
    transform = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    # We only need test data here, since we assume that parameters have been selected
    test_dataset = openset_imagenet.ImagenetDataset(
        csv_file=cfg.data.test_file.format(args.protocol),
        imagenet_path=cfg.data.imagenet_path,
        transform=transform)

    # Info on console
    print("\n========== Data ==========")
    print(f"Test dataset len:{len(test_dataset)}, labels:{test_dataset.label_count}")

    # create data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers)

    # create device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
    else:
        print("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    if args.loss == "garbage":
        n_classes = test_dataset.label_count - 1 # we use one class for the negatives; the dataset has two additional  labels: negative and anknown
    else:
        n_classes = test_dataset.label_count - 2  # number of classes - 2 when training was without garbage class

    # create model
    suffix = "best" if args.use_best else "curr"

    if args.algorithm != "threshold":
        opt = cfg.optimized[args.algorithm]
        popt = opt[f"p{args.protocol}"][args.loss]
    if args.algorithm == 'proser':
        base = openset_imagenet.ResNet50(
            fc_layer_dim=n_classes,
            out_features=n_classes,
            logit_bias=False)

        model = openset_imagenet.model.ResNet50Proser(
            resnet_base = base,
            dummy_count = popt.dummy_count,
            fc_layer_dim=n_classes)

        model_path = opt.output_model_path.format(output_directory, args.loss, args.algorithm, popt.epochs, popt.dummy_count, suffix)
    else:
        model = openset_imagenet.ResNet50(
            fc_layer_dim=n_classes,
            out_features=n_classes,
            logit_bias=False)

        model_path = cfg.model_path.format(output_directory, args.loss, "threshold", suffix)

    start_epoch, best_score = openset_imagenet.train.load_checkpoint(model, model_path)
    print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)

    if args.algorithm in ["openmax", "evm"]:
        # load EVM or OpenMax model
        if args.algorithm == "openmax":
            key = get_param_string("openmax", tailsize=popt.tailsize, distance_multiplier=popt.distance_multiplier)
        else:
            key = get_param_string("evm", tailsize=popt.tailsize, distance_multiplier=popt.distance_multiplier, cover_threshold=opt.cover_threshold)
        model_path = opt.output_model_path.format(output_directory, args.loss, args.algorithm, key, opt.distance_metric)
        model_dict = pickle.load(open(model_path, "rb"))


    # Test Section
    print("getting features and logits for test set .... ")

    if args.algorithm == 'proser':
        gt, logits, features, scores = openset_imagenet.proser.get_arrays(
            model=model,
            loader=test_loader
        )
    else:
        gt, logits, features, scores = openset_imagenet.train.get_arrays(
            model=model,
            loader=test_loader,
            garbage=args.loss=="garbage"
        )

    if args.algorithm == 'openmax':
        #scores are being adjusted her through openmax alpha
        print("adjusting probabilities for openmax with alpha")
        hyperparams = openset_imagenet.util.NameSpace(dict(distance_metric = opt.distance_metric))
        scores = compute_adjust_probs(gt, logits, features, scores, model_dict, "openmax", args.gpu, hyperparams, alpha_index=0)
    elif args.algorithm == 'evm':
        print("computing probabilities for evm")
        hyperparams = openset_imagenet.util.NameSpace(dict(distance_metric = opt.distance_metric))
        scores = compute_probs(gt, logits, features, scores, model_dict, "evm", args.gpu, hyperparams)

    file_path = Path(output_directory) / f"{args.loss}_{args.algorithm}_test_arr_{suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")


if __name__=='__main__':
    main()
