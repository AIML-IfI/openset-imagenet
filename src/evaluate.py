""" Independent code for inference in testing dataset. The functions are included and executed
in the train.py script."""
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms as tf
from torch.utils.data import DataLoader
from dataset import ImagenetDataset
from train import load_checkpoint
from model import ResNet50


def get_args():
    """Gets the evaluation parameters."""
    parser = argparse.ArgumentParser("Get parameters for evaluation")

    # directory parameters
    parser.add_argument(
        "--imagenet_dir",
        type=Path,
        default=Path(r"/local/scratch/datasets/ImageNet/ILSVRC2012/"),
        help="Imagenet root directory")
    parser.add_argument(
        "--name",
        type=str,
        help="Name of current experiment, used for naming logs and checkpoints")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory. Default is the same directory as the script")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint .pth file")
    parser.add_argument(
        "--val_file",
        type=Path,
        default=Path(__file__).parent / "validation.csv",  # TODO: default is wrong
        help="Path to validation file")
    parser.add_argument(
        "--test_file",
        type=str,
        default=Path(__file__).parent / "test.csv",
        help="Path to test file")
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Default: 32")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Data loaders number of workers, default:4")
    parser.add_argument(
        "--loss",
        default="entropic",
        help="[objectosphere, entropic, softmax, BGsoftmax]")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold tau: if max(S_c(X))>tau: class=c o/w x is unknown")
    return parser.parse_args()


def get_arrays(eval_model, loader, eval_device, batch_size):
    """ Extract deep features, logits and targets for all dataset. Returns numpy arrays

    Args:
        eval_model (torch model): Model.
        loader (torch dataloader): Data loader.
        eval_device (cuda): Cuda id.
        arg: Arguments structure.
    """
    eval_model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        logits_dim = eval_model.logits.out_features  # logits output classes
        features_dim = eval_model.net.fc.out_features  # features dimensionality
        all_targets = torch.empty(data_len, device=eval_device)  # store all targets
        all_logits = torch.empty((data_len, logits_dim), device=eval_device)   # store all logits
        all_feat = torch.empty((data_len, features_dim), device=eval_device)   # store all features
        all_scores = torch.empty((data_len, logits_dim), device=eval_device)

        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            curr_b_size = labels.shape[0]  # current batch size, very last batch has different value
            images = images.to(eval_device)
            labels = labels.to(eval_device)
            logit, feature = eval_model(images, features=True)
            score = torch.nn.functional.softmax(logit, dim=1)
            # accumulate results in all_tensor
            index = i * batch_size
            all_targets[index:index + curr_b_size] = labels
            all_logits[index:index + curr_b_size] = logit
            all_feat[index:index + curr_b_size] = feature
            all_scores[index:index + curr_b_size] = score
        return(
            all_targets.detach().cpu().numpy(),
            all_logits.detach().cpu().numpy(),
            all_feat.detach().cpu().numpy(),
            all_scores.detach().cpu().numpy())


if __name__ == "__main__":
    args = get_args()

    # Create transformations
    transform_val = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    # create datasets
    val_dataset = ImagenetDataset(
        csv_file=args.val_file,
        imagenet_path=args.imagenet_dir,
        transformation=transform_val)

    test_dataset = ImagenetDataset(
        csv_file=args.test_file,
        imagenet_path=args.imagenet_dir,
        transformation=transform_val)

    # Info on console
    print("\n========== Data ==========")
    print(f"Val dataset len:{len(val_dataset)}, labels:{val_dataset.label_count}")
    print(f"Test dataset len:{len(test_dataset)}, labels:{test_dataset.label_count}")

    # create directory if not exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # create data loaders
    val_loader = DataLoader(val_dataset, batch=args.batch, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch=args.batch, num_workers=args.workers)

    # create device
    device = torch.device("cuda")

    # if args.loss in ['softmax', 'BGsoftmax']:
    #     n_classes = val_ds.label_cnt
    #     print('is softmax', n_classes)
    # else:
    #     n_classes = val_ds.label_cnt - 1

    if val_dataset.has_negatives():
        n_classes = val_dataset.label_count - 1  # number of classes - 1 when training with unknowns
    else:
        n_classes = val_dataset.label_count

    # create model
    model = ResNet50(fc_layer_dim=n_classes, out_features=n_classes, logit_bias=False)
    start_epoch, best_score = load_checkpoint(args.checkpoint, model)
    model.to(device)

    print("========== Evaluating ==========")
    print("Validation data:")
    # extracting arrays for validation
    gt, logits, features, scores = get_arrays(
        eval_model=model,
        loader=val_loader,
        eval_device=device,
        batch_size=args.batch)
    file_path = args.out_dir / f"{args.name}_val_arr.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")

    # extracting arrays for test
    print("Test data:")
    gt, logits, features, scores = get_arrays(
        eval_model=model,
        loader=test_loader,
        eval_device=device,
        batch_size=args.batch)
    file_path = args.out_dir / f"{args.name}_test_arr.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")
