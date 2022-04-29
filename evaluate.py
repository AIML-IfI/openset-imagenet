import random
import torch
import argparse
import numpy as np
from pathlib import Path
from torchvision import transforms as tf
from dataset import Imagenet_dataset
from torch.utils.data import DataLoader
from model import ResNet50
from tqdm import tqdm
from collections import OrderedDict


def get_args():
    """Gets the hyperparameters """
    parser = argparse.ArgumentParser('Hyperparameters')

    # directory parameters
    parser.add_argument('--imagenet_path', default=Path(r'/local/scratch/datasets/ImageNet/ILSVRC2012/'),
                        type=Path, help='Imagenet directory', metavar='')
    parser.add_argument('--exp_name', default='', type=str, metavar='',
                        help='Name of current experiment, used for naming file of logs and checkpoints')
    parser.add_argument('--output_dir', default=Path(__file__).parent, type=Path, metavar='',
                        help='output directory to save arrays. Default is the same directory as the script')
    parser.add_argument('--checkpoint', type=Path, help='Path to saved checkpoint .pth file', metavar='')
    parser.add_argument('--val_file', default=Path(__file__).parent/'validation.csv', type=Path, metavar='',
                        help='path to validation file')
    parser.add_argument('--test_file', default=Path(__file__).parent/'test.csv', type=str, metavar='',
                        help='path to test file')
    # common parameters
    parser.add_argument('--batch_size', default=32, type=int, help='Default: 32', metavar='')
    parser.add_argument('--workers', default=4, type=int, metavar='',
                        help='Data loaders number of workers, default:4')
    parser.add_argument('--seed', default=343443, type=int, help='seed default 343443', metavar='')
    parser.add_argument('--loss', default='objectosphere', metavar='',
                        help='[objectosphere, entropic, softmax]')
    parser.add_argument('-t', '--threshold', default=0.8, type=float, metavar='',
                        help='Threshold tau for predicting: if max(S_c(X))>tau: class=c o/w x is unknown')
    parser.add_argument('--gpu', default=0, type=int, metavar='',
                        help='index of gpu to use for evaluation, default: 0')
    args = parser.parse_args()
    # Directory Parameters
    return args


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_checkpoint(file_path, model, opt=None):
    """Loads a checkpoint in CPU. If the model was saved using DistributedDataParallel, removes the
    word 'module' from the state_dictionary keys to load it in a single device"""
    checkpoint = torch.load(file_path, map_location='cpu')
    key = list(checkpoint['model_state_dict'].keys())[0]
    if key[:6] == 'module':
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            key = k[7:]  # remove 'module'
            new_state_dict[key] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    print('epoch', checkpoint['epoch'])
    del checkpoint
    print('Loaded model from: {}'.format(file_path))
    return model


def get_arrays(model, loader, device, args):
    """Extract deep features, logits and targets for all dataset.
    Returns numpy arrays"""
    model.eval()
    with torch.no_grad():
        N = len(loader.dataset)         # dataset length
        C = model.logits.out_features   # logits output classes
        F = model.net.fc.out_features       # features dimensionality
        all_targets = torch.empty(N, device=device)  # store all targets
        all_logits = torch.empty((N, C), device=device)   # store all logits
        all_feat = torch.empty((N, F), device=device)   # store all features
        all_scores = torch.empty((N, C), device=device)

        for i, (x, t) in tqdm(enumerate(loader), total=len(loader)):
            n = t.shape[0]  # current batch size, very last batch has different value
            x = x.to(device)
            t = t.to(device)
            logits, features = model(x, features=True)
            scores = torch.nn.functional.softmax(logits, dim=1)
            # accumulate resutls in all_tensor
            ix = i*args.batch_size
            all_targets[ix:ix+n] = t
            all_logits[ix:ix+n] = logits
            all_feat[ix:ix+n] = features
            all_scores[ix:ix+n] = scores
        return(
            all_targets.detach().cpu().numpy(),
            all_logits.detach().cpu().numpy(),
            all_feat.detach().cpu().numpy(),
            all_scores.detach().cpu().numpy()
        )


if __name__ == '__main__':
    args = get_args()
    set_seeds(args.seed)
    # Create transformations
    val_tf = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor(),
         #  tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]
        )
    # create datasets
    val_ds = Imagenet_dataset(args.val_file, args.imagenet_path, val_tf)
    test_ds = Imagenet_dataset(args.test_file, args.imagenet_path, val_tf)

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
    device = torch.device('cuda:{}'.format(args.gpu))
    # device = torch.device("cpu")

    if args.loss == 'softmax':
        n_classes = val_ds.label_cnt
        print('is softmax', n_classes)
    else:
        n_classes = val_ds.label_cnt - 1

    # create model
    model = ResNet50(fc_layer_dim=n_classes, out_features=n_classes, logit_bias=False)
    model = load_checkpoint(args.checkpoint, model)
    model.to(device)

    print('========== Evaluating ==========')
    print('Validation data:')
    # extracting arrays for validation
    gt, logits, features, scores = get_arrays(model, val_loader, device, args)
    file_path = args.output_dir/'{}_val_arr.npz'.format(args.exp_name)
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
    file_path = args.output_dir/'{}_test_arr.npz'.format(args.exp_name)
    np.savez(
        file_path,
        gt=gt,
        logits=logits,
        features=features,
        scores=scores
    )
    print('ground truth, logits, deep features, softmax scores saved in: {}'.format(file_path))

    # # Extract the values - validation
    # pred, score, gt, feature = predict_val(model, val_loader, device, args)
    # np.savez(
    #     args.output_dir/'{}_val_arr.npz'.format(args.exp_name),
    #     pred=pred,
    #     score=score,
    #     gt=gt,
    #     feature=feature
    #     )

    # # Extract the values - test
    # pred, score, gt, feature = predict_val(model, test_loader, device, args)
    # np.savez(
    #     args.output_dir/'{}_test_arr.npz'.format(args.exp_name),
    #     pred=pred,
    #     score=score,
    #     gt=gt,
    #     feature=feature
    #     )

    # Correct classification rate: Total number of samples in:
    # Dc -> Cardinality of samples of known classes
    # Db -> Cardinality of known unknown samples
    # Da -> Cardinality of unknown unknown samples
    # Du -> All unknown samples (Db U Da)
    # Dc = np.sum(gt > -1)
    # Du = np.sum(gt < 0)
    # Db = np.sum(gt == -1)
    # Da = np.sum(gt == -2)
    # ccr, fpr_Du, fpr_Db, fpr_Da = [], [], [], []
    # for theta in np.unique(score):
    #     corr_count = np.count_nonzero((gt > -1)*(pred == gt)*(score >= theta))
    #     corr_count /= Dc
    #     ccr.append(corr_count)
    #     count_u = np.count_nonzero((gt < 0)*(pred >= 0)*(score >= theta))
    #     count_u /= Du
    #     fpr_Du.append(count_u)
    #     count_b = np.count_nonzero((gt == -1)*(pred >= 0)*(score >= theta))
    #     count_b /= Db
    #     fpr_Db.append(count_b)
    #     count_a = np.count_nonzero((gt == -2)*(pred >= 0)*(score >= theta))
    #     count_a /= Da
    #     fpr_Da.append(count_a)
    # print('{:.5f},{:.5f},{:.5f},{:.5f}'.format(corr_count, count_u, count_b, count_a))

    # # replace -2 labels to -1 to get binary AUC
    # gt[gt == -2] = -1
    # auc = metrics.auc_score_binary(gt, score)
    # print('auc', auc)

    # # plot in multipage pdf
    # with PdfPages(args.data_dir/'figures.pdf') as pdf:
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #     ax.plot(fpr_Du, ccr, label='Du: Unk All')
    #     ax.plot(fpr_Da, ccr, label='Da: Unk Unk')
    #     ax.plot(fpr_Db, ccr, label='Db: Knw Unk')
    #     ax.set_xlabel('False Positive Rate', fontsize=14)
    #     ax.set_ylabel('Correct Classification Rate', fontsize=14)
    #     ax.set_title('CCR')
    #     ax.tick_params(bottom=True, top=True, left=True, right=True, direction='inout')
    #     ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=13)
    #     plt.legend(frameon=False)
    #     pdf.savefig()
    #     plt.close()

    #     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #     ax.hist(norms, bins=500, log=True)
    #     pdf.savefig()
    #     plt.close()
