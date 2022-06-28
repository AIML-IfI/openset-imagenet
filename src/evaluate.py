"""Independent code for inference in testing dataset. The functions are included
and executed in the train.py script."""
import torch
import argparse
import numpy as np
from pathlib import Path
from torchvision import transforms as tf
from dataset import ImagenetDataset
from torch.utils.data import DataLoader
from model import ResNet50
from tqdm import tqdm
from collections import OrderedDict


def get_args():
    """Gets the hyperparameters """
    parser = argparse.ArgumentParser('Hyperparameters')

    # directory parameters
    parser.add_argument('--imagenet_path',
                        default=Path(r'/local/scratch/datasets/ImageNet/ILSVRC2012/'),
                        type=Path, help='Imagenet directory', metavar='')
    parser.add_argument('--exp_name', default='', type=str, metavar='',
                        help='Name of current experiment, used for naming logs and checkpoints')
    parser.add_argument('--output_dir', default=Path(__file__).parent, type=Path, metavar='',
                        help='output directory. Default is the same directory as the script')
    parser.add_argument('--checkpoint', type=Path, help='Path to saved checkpoint .pth file',
                        metavar='')
    parser.add_argument('--val_file', default=Path(__file__).parent / 'validation.csv', type=Path,
                        metavar='', help='path to validation file')
    parser.add_argument('--test_file', default=Path(__file__).parent / 'test.csv', type=str,
                        metavar='', help='path to test file')
    # common parameters
    parser.add_argument('--batch_size', default=32, type=int, help='Default: 32', metavar='')
    parser.add_argument('--workers', default=4, type=int, metavar='',
                        help='Data loaders number of workers, default:4')
    parser.add_argument('--loss', default='objectosphere', metavar='',
                        help='[objectosphere, entropic, softmax, BGsoftmax]')
    parser.add_argument('-t', '--threshold', default=0.5, type=float, metavar='',
                        help='Threshold tau: if max(S_c(X))>tau: class=c o/w x is unknown')
    return parser.parse_args()


def load_checkpoint(filepath, model_):
    """Loads a checkpoint in CPU. If the model was saved using DistributedDataParallel, removes the
    word 'module' from the state_dictionary keys to load it in a single device"""
    checkpoint = torch.load(filepath, map_location='cpu')
    key = list(checkpoint['model_state_dict'].keys())[0]
    if key[:6] == 'module':
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            key = k[7:]  # remove 'module'
            new_state_dict[key] = v
        model_.load_state_dict(new_state_dict)
    else:
        model_.load_state_dict(checkpoint['model_state_dict'])
    print('epoch', checkpoint['epoch'])
    del checkpoint
    print('Loaded model from: {}'.format(filepath))
    return model_


def get_arrays(model_, loader, device_, args_):
    """Extract deep features, logits and targets for all dataset.
    Returns numpy arrays"""
    model_.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        logits_dim = model_.logits.out_features  # logits output classes
        features_dim = model_.net.fc.out_features  # features dimensionality
        all_targets = torch.empty(data_len, device=device_)  # store all targets
        all_logits = torch.empty((data_len, logits_dim), device=device_)   # store all logits
        all_feat = torch.empty((data_len, features_dim), device=device_)   # store all features
        all_scores = torch.empty((data_len, logits_dim), device=device_)

        for i, (x, t) in tqdm(enumerate(loader), total=len(loader)):
            n = t.shape[0]  # current batch size, very last batch has different value
            x = x.to(device_)
            t = t.to(device_)
            logits_, features_ = model(x, features=True)
            scores_ = torch.nn.functional.softmax(logits_, dim=1)
            # accumulate results in all_tensor
            ix = i * args_.batch_size
            all_targets[ix:ix + n] = t
            all_logits[ix:ix + n] = logits_
            all_feat[ix:ix + n] = features_
            all_scores[ix:ix + n] = scores_
        return(
            all_targets.detach().cpu().numpy(),
            all_logits.detach().cpu().numpy(),
            all_feat.detach().cpu().numpy(),
            all_scores.detach().cpu().numpy()
        )


if __name__ == '__main__':
    args = get_args()
    # Create transformations
    val_tf = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()]
    )
    # create datasets
    val_ds = ImagenetDataset(args.val_file, args.imagenet_path, val_tf)
    test_ds = ImagenetDataset(args.test_file, args.imagenet_path, val_tf)

    # Info on console
    print('\n========== Data ==========')
    print('val_ds len:{}, labels:{}'.format(len(val_ds), val_ds.label_cnt))
    print('test_ds len:{}, labels:{}'.format(len(test_ds), test_ds.label_cnt))

    # create directory if not exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # create data loaders
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)

    # create device
    device = torch.device('cuda')

    # if args.loss in ['softmax', 'BGsoftmax']:
    #     n_classes = val_ds.label_cnt
    #     print('is softmax', n_classes)
    # else:
    #     n_classes = val_ds.label_cnt - 1

    if val_ds.has_unknowns():
        n_classes = val_ds.label_cnt - 1  # number of classes - 1 when training with unknowns
    else:
        n_classes = val_ds.label_cnt

    # create model
    model = ResNet50(fc_layer_dim=n_classes, out_features=n_classes, logit_bias=False)
    model = load_checkpoint(args.checkpoint, model)
    model.to(device)

    print('========== Evaluating ==========')
    print('Validation data:')
    # extracting arrays for validation
    gt, logits, features, scores = get_arrays(model, val_loader, device, args)
    file_path = args.output_dir / '{}_val_arr.npz'.format(args.exp_name)
    np.savez(
        file_path,
        gt=gt,
        logits=logits,
        features=features,
        scores=scores
    )
    print('ground truth, logits, deep features, softmax scores saved in: {}'.format(file_path))

    # extracting arrays for test
    print('Test data:')
    gt, logits, features, scores = get_arrays(model, test_loader, device, args)
    file_path = args.output_dir / '{}_test_arr.npz'.format(args.exp_name)
    np.savez(
        file_path,
        gt=gt,
        logits=logits,
        features=features,
        scores=scores
    )
    print('ground truth, logits, deep features, softmax scores saved in: {}'.format(file_path))
