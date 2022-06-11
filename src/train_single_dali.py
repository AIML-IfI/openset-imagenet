import random
import time
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
import metrics
import adversary
from dataset import ImagenetDataset
from model import ResNet50
from losses import AverageMeter, EarlyStopping, EntropicLoss, ObjectoLoss
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.optim import lr_scheduler
from vast.tools import set_device_gpu
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from sys import stderr
import hydra
from omegaconf import DictConfig
from os import getcwd
from pipe import create_dali_pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import pandas as pd
# Global objects
best_score = 0.0
start_epoch = 0


def check_config(cfg):
    # Checks
    if (cfg.adv.who != 'no_adv') and cfg.dist.distributed:
        m = "Can't train adversarial samples in distributed mode. Try single GPU"
        raise Exception(m)


def set_seeds(seed):
    """ Set a seed for randomness in different modules e.g. torch, random, numpy
    Args:
        seed (int): Seed
    """

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    """Saves current state dictionary into a checkpoint.
    Args:
        f_name: File path to save
        model (nn.Module): Model to save
        epoch (int): Current epoch
        opt (torch optimiser): Current optimiser
        best_score_ (float): Current best score on validation set
        scheduler (torch lr_scheduler, optional): Current scheduler. Defaults to None.
    """

    # If model is distributed extracts the module.
    state = model.state_dict()
    data = {
        'epoch': epoch+1,
        'model_state_dict': state,
        'opt_state_dict': opt.state_dict(),
        'best_score': best_score_
        }
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()
    torch.save(data, f_name)


def load_checkpoint(model, ckpt_path, opt=None, device='cpu', scheduler=None):
    """ Loads a checkpoint in CPU by default. If the model was saved using DistributedDataParallel, removes
    the word 'module' from the state_dictionary keys to load it in a single device.
    If fine-tuning model then optimizer should be none to start from clean optimizer parameters.

    Args:
        model (torch nn.module): Requires a model to load the state dictionary
        ckpt_path (Path): File path
        opt (torch optimizer, optional): An optimizer to load the state dictionary. Defaults to None.
        device (str, optional): Device to load the checkpoint, can be loaded directly to a cuda device.
                                Defaults to 'cpu'.
        scheduler (torch lr_scheduler, optional): Learning rate scheduler. Defaults to None.
    """

    global best_score
    global start_epoch
    file_path = Path(ckpt_path)
    if file_path.is_file:   # First check if file exists
        checkpoint = torch.load(file_path, map_location=device)
        key = list(checkpoint['model_state_dict'].keys())[0]
        # If module was saved as DistributedDataParallel then removes the world 'module' from dictionary keys
        if key[:6] == 'module':
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                key = k[7:]  # remove 'module'
                new_state_dict[key] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        # Load optimiser state
        if opt is not None:
            opt.load_state_dict(checkpoint['opt_state_dict'])
        start_epoch = checkpoint['epoch']
        # Load scheduler state
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        # Load best score
        if 'best_score' in checkpoint:
            best_score = checkpoint['best_score']
            logger.info('best score of loaded model: {:.3f}'.format(checkpoint['best_score']))
        logger.info('loaded {} at epoch {}'.format(file_path, checkpoint['epoch']))
        del checkpoint
    else:
        logger.info('Checkpoint file not found')


def predict(scores, threshold):
    pred_score, pred_class = torch.max(scores, dim=1)
    unk = (pred_score < threshold)
    pred_class[unk] = -1
    return torch.stack((pred_class, pred_score), dim=1)


def filter_correct(logits, target, threshold, features=None):
    """Returns the indices of correctly predicted known samples.
    Args:
        logits (tensor): Logits tensor
        target (tensor): Targets tensor
        threshold (float): Minimum score for the target to be classified as known.
        features (tensor, optional): Features tensor. Defaults to None.
    Returns:
        tuple: Tuple containing in fist position the tensor with indices of correctly predicted samples.
    """
    with torch.no_grad():
        if features is None:
            scores = torch.nn.functional.softmax(logits, dim=1)
            pred = predict(scores, threshold)
            correct = (target >= 0) * (pred[:, 0] == target)
            idx = torch.nonzero(correct, as_tuple=True)
            return idx


def train(model, data_loader, optimizer, device, loss_fn, trackers, cfg):
    # reset dictionary of metrics
    for t in trackers.values():
        t.reset()
    j = None
    for data in data_loader:
        x = data[0]["data"]
        t = data[0]["label"].squeeze(-1).long()
        # data_time = time.time() - train_time
        model.train()
        n = t.shape[0]  # Samples in c batch
        optimizer.zero_grad(set_to_none=True)
        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        if cfg.adv.who == 'fgsm':    # Tracking the gradient w.r.t input
            x.requires_grad_()
            x.grad = None

        logits, features = model(x, features=True)

        if cfg.loss.type == 'objectosphere':
            j = loss_fn(features, logits, t, cfg.loss.alpha)
            trackers['j_o'].update(loss_fn.objecto_value, n)
            trackers['j_e'].update(loss_fn.entropic_value, n)
        elif cfg.loss.type == 'entropic':
            j = loss_fn(logits, t)
            # trackers['j_e'].update(j.item(), n)
        elif cfg.loss.type == 'softmax':
            j = loss_fn(logits, t)
            trackers['j_s'].update(j.item(), n)
        j.backward()
        if cfg.adv.who == 'no_adv':
            optimizer.step()
        else:
            # for adversarial: filter samples, create adv_samples, calculate loss, backward pass
            model.eval()  # To avoid updates in current batch normalisation layers

            # Get the candidates to adversarial samples
            if cfg.adv.mode == 'filter':
                correct_idx = filter_correct(logits, t, cfg.threshold, features=None)
                # print('idx_crr_device', correct_idx[0].get_device())
                num_adv_samples = len(correct_idx[0])
            elif cfg.adv.mode == 'full':
                correct_idx = torch.arange(n, requires_grad=False, device=device)
                num_adv_samples = len(correct_idx)
                # print('device', correct_idx.get_device(), 'adv samples', num_adv_samples)

            trackers['num_adv'].update(num_adv_samples)

            if num_adv_samples > 0:
                # print('adversarial samples', num_adv_samples)
                x_corr = x[correct_idx]
                # Create the adversarial sample based on who is the adversary
                if cfg.adv.who == 'gaussian':
                    x_adv, t_adv = adversary.add_gaussian_noise(x_corr, loc=0, std=cfg.adv.std, device=device)
                elif cfg.adv.who == 'random':
                    x_adv, t_adv = adversary.add_random_noise(x_corr, epsilon=cfg.adv.epsilon, device=device)
                elif cfg.adv.who == 'fgsm':
                    x_corr_grad = x.grad[correct_idx]
                    x_adv, t_adv = adversary.fgsm_attack(x_corr, cfg.adv.epsilon, x_corr_grad, device=device)
                else:
                    print('skipping adversarials')
                    continue
                # print('x_adv', x_adv.get_device(), 't_adv', t_adv.get_device())

                # forward pass with adversarial samples
                logits, features = model(x_adv)
                if cfg.loss.type == 'objectosphere':
                    j = loss_fn(features, logits, t_adv, cfg.loss.alpha)
                elif cfg.loss.type == 'entropic':
                    j = loss_fn(logits, t_adv)
                # print('j adv', j.get_device())
                trackers['j_adv'].update(j.item(), num_adv_samples)
                j.backward()
            optimizer.step()


def validate(model, loader, device, loss_fn, n_classes, trackers, cfg):
    for t in trackers.values():
        t.reset()

    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)  # size of dataset
        all_t = torch.empty(data_len, device=device).detach()  # store all targets
        all_scores = torch.empty((data_len, n_classes), device=device).detach()  # store all scores

        for i, (x, t) in enumerate(loader):
            n = t.shape[0]  # current batch size, last batch has different value
            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            logits, features = model(x)
            scores = torch.nn.functional.softmax(logits, dim=1)

            if cfg.loss.type == 'objectosphere':
                j = loss_fn(features, logits, t, cfg.loss.alpha)
                trackers['j_o'].update(loss_fn.objecto_value, n)
                trackers['j_e'].update(loss_fn.entropic_value, n)
            elif cfg.loss.type == 'entropic':
                j = loss_fn(logits, t)
                trackers['j_e'].update(j.item(), n)
            elif cfg.loss.type == 'softmax':
                j = loss_fn(logits, t)
                trackers['j_s'].update(j.item(), n)
            # accumulate partial results in empty tensors
            ix = i*cfg.batch_size
            all_t[ix:ix+n] = t
            all_scores[ix:ix+n] = scores

            # average confidence tracking
            conf = metrics.confidence(scores, t, 1 / n_classes)
            trackers['conf'].update((conf[0]/n).item(), n)  # average confidence

        # Validate using AUC
        if cfg.loss.type == 'softmax':
            auc = metrics.auc_score_multiclass(all_t, all_scores)
        else:
            max_score, _ = torch.max(all_scores, dim=1)
            # print('max_score shape {}, t shape {}'.format(max_score.shape, all_t.shape))
            auc = metrics.auc_score_binary(all_t, max_score)
        trackers['auc'].update(auc, data_len)


def save_eval_arrays(model, loader, device, batch_size, file_name):
    """Extract deep features, logits and targets for all dataset.
    Returns numpy arrays"""
    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        logits_dim = model.logits.out_features   # logits output classes
        features_dim = model.net.fc.out_features   # features dimensionality
        all_targets = torch.empty(data_len, device=device)      # store all targets
        all_logits = torch.empty((data_len, logits_dim), device=device)  # store all logits
        all_feat = torch.empty((data_len, features_dim), device=device)    # store all features
        all_scores = torch.empty((data_len, logits_dim), device=device)  # store all scores

        for i, (x, t) in enumerate(loader):
            n = t.shape[0]  # current batch size, very last batch has different value
            x = x.to(device)
            t = t.to(device)
            logits, features = model(x, features=True)
            scores = torch.nn.functional.softmax(logits, dim=1)
            # accumulate results in all_tensor
            ix = i*batch_size
            all_targets[ix:ix+n] = t
            all_logits[ix:ix+n] = logits
            all_feat[ix:ix+n] = features
            all_scores[ix:ix+n] = scores
    # Parse to numpy arrays
    gt = all_targets.detach().cpu().numpy()
    logits = all_logits.detach().cpu().numpy()
    features = all_feat.detach().cpu().numpy()
    scores = all_scores.detach().cpu().numpy()
    # Save arrays to file
    np.savez(file_name, gt=gt, logits=logits, features=features, scores=scores)
    logger.info('gt, logits, features, scores saved in: {}'.format(file_name))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # # Setting the logger
    msg_format = "{time:DD_MM_HH:mm} {message}"
    logger.configure(handlers=[{"sink": stderr, "level": "INFO", "format": msg_format}])
    logger.add(cfg.log_name, format=msg_format, level="INFO", mode='w')
    check_config(cfg)

    worker(cfg)


def worker(cfg):
    # referencing best score and setting seeds
    global best_score
    set_seeds(cfg.seed)

    # Set image transformations
    train_tf = tf.Compose(
        [
         tf.Resize(256),
         tf.RandomCrop(224),
         tf.RandomHorizontalFlip(0.5),
         tf.ToTensor()]
    )

    val_tf = tf.Compose([tf.Resize(256), tf.CenterCrop(224), tf.ToTensor()])
    # val_tf = tf.Compose([tf.CenterCrop((224)), tf.ToTensor()])

    # create datasets
    data_dir = Path(cfg.data.data_dir)
    train_file = data_dir/cfg.data.train_file
    val_file = data_dir/cfg.data.val_file

    if train_file.exists() and val_file.exists():
        train_ds = ImagenetDataset(train_file, cfg.data.imagenet_path, train_tf)
        val_ds = ImagenetDataset(val_file, cfg.data.imagenet_path, val_tf)
    else:
        raise FileNotFoundError('train/validation file does not exist')
    # print(cfg.workers, type(cfg.workers))
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=cfg.workers,
    #     pin_memory=True,
    #     prefetch_factor=2,
    #     )

    t_file = pd.read_csv(train_file, header=None)
    img_paths = t_file.iloc[:, 0].tolist()
    img_labels = t_file.iloc[:, 1].tolist()

    dali_cpu = False
    pipe = create_dali_pipeline(
        batch_size=cfg.batch_size,
        num_threads=cfg.workers,
        device_id=0,
        seed=343443,
        data_dir=cfg.data.imagenet_path,
        im_paths=img_paths,
        im_labels=img_labels,
        crop=224,
        size=224,
        dali_cpu=dali_cpu,
        is_training=True
        )
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True
        )

    # Callbacks
    if cfg.patience > 0:
        early_stopping = EarlyStopping(patience=cfg.patience)

    # setup device
    device = torch.device('cuda:0')
    set_device_gpu(index=0)

    # Set dictionaries to keep track of the losses
    t_metrics = defaultdict(AverageMeter)
    v_metrics = defaultdict(AverageMeter)

    if train_ds.has_unknowns():
        n_classes = train_ds.label_cnt - 1  # number of classes - 1 when training with unknowns
    else:
        n_classes = train_ds.label_cnt
    if cfg.loss.type == 'objectosphere':
        loss = ObjectoLoss(n_classes, cfg.loss.w, cfg.loss.xi)
    elif cfg.loss.type == 'entropic':
        loss = EntropicLoss(n_classes, cfg.loss.w)
    elif cfg.loss.type == 'softmax':
        loss = torch.nn.CrossEntropyLoss().to(device)

    # Create the model
    model = ResNet50(fc_layer_dim=n_classes, out_features=n_classes, logit_bias=False)
    model.to(device)

    # Create optimizer
    if cfg.opt.type == 'sgd':
        opt = torch.optim.SGD(params=model.parameters(), lr=cfg.opt.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(params=model.parameters(), lr=cfg.opt.lr)

    # Learning rate scheduler
    if cfg.opt.decay > 0:
        scheduler = lr_scheduler.StepLR(opt, step_size=cfg.opt.decay, gamma=cfg.opt.gamma, verbose=True)
    else:
        scheduler = None

    # Resume a training from a checkpoint
    if cfg.checkpoint is not None:
        # Get the relative path of the checkpoint wrt train.py
        original_wd = Path(hydra.utils.get_original_cwd())
        checkpoint_path = original_wd/cfg.checkpoint
        if cfg.train_mode == 'finetune':
            load_checkpoint(model, checkpoint_path, None, device, None)
            logger.info('Fine tuning: Setting best score to 0')
            best_score = 0
        else:
            load_checkpoint(model, checkpoint_path, opt, device, scheduler)

    start_epsilon = cfg.adv.epsilon  # Needed to keep initial epsilon when using decay
    adv_who = cfg.adv.who           # Store initial adversary
    # Checks if train the model without adversarials for a nr. of epochs then add adversarial samples.
    if cfg.adv.wait > 0:
        cfg.adv.who = 'no_adv'

    # Set the final epoch
    global start_epoch
    cfg.epochs = start_epoch + cfg.epochs
    writer = SummaryWriter(log_dir=getcwd())    # summary writer

    # Info on console
    logger.info('========== Datasets ==========')
    logger.info('train_ds len:{}, labels:{}'.format(len(train_ds), train_ds.label_cnt))
    logger.info('val_ds len:{}, labels:{}'.format(len(val_ds), val_ds.label_cnt))
    logger.info('Total batch size: {}'.format(cfg.batch_size*cfg.dist.gpus))
    logger.info('========== Training info ==========')
    logger.info('Initial epoch: {}'.format(start_epoch))
    logger.info('Last epoch: {}'.format(cfg.epochs))
    logger.info('General mode: {}'.format(cfg.train_mode))
    logger.info('Batch size: {}'.format(cfg.batch_size))
    logger.info('workers: {}'.format(cfg.workers))
    logger.info('Adversary: {}'.format(cfg.adv.who))
    logger.info('Adversary mode: {}'.format(cfg.adv.mode))
    logger.info('Loss: {}'.format(cfg.loss.type))
    logger.info('Optimiser: {}'.format(cfg.opt.type))
    logger.info('Learning rate: {}'.format(cfg.opt.lr))
    logger.info('Device: {}'.format(device))
    logger.info('Training...')

    for epoch in range(start_epoch, cfg.epochs):
        epoch_time = time.time()

        if (cfg.adv.wait > 0) and (epoch >= cfg.adv.wait):
            cfg.adv.who = adv_who

        # calculate epsilon
        if (cfg.adv.who in ['fgsm', 'random']) and 0 < cfg.adv.mu < 1 and cfg.adv.decay > 0:
            cfg.adv.epsilon = adversary.decay_epsilon(
                start_eps=start_epsilon,
                mu=cfg.adv.mu,
                curr_epoch=epoch,
                wait_epochs=cfg.adv.decay
                )
            logger.info('epsilon:{:.4f}'.format(cfg.adv.epsilon))

        # training loop
        train(model, train_loader, optimizer=opt, device=device, loss_fn=loss, trackers=t_metrics, cfg=cfg)
        train_time = time.time() - epoch_time

        # validation loop
        validate(model, val_loader, device, loss_fn=loss, n_classes=n_classes, trackers=v_metrics, cfg=cfg)
        curr_score = v_metrics['auc'].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        # Logging metrics to tensorboard object
        if cfg.loss.type == 'objectosphere':
            writer.add_scalar('loss/objecto', t_metrics['j_o'].avg, epoch)
            writer.add_scalar('loss/entropic', t_metrics['j_e'].avg, epoch)
            writer.add_scalar('val/objecto', v_metrics['j_o'].avg, epoch)
            writer.add_scalar('val/entropic', v_metrics['j_e'].avg, epoch)
            if cfg.adv.who != 'no_adv':
                writer.add_scalar('loss/adversarial', t_metrics['j_adv'].avg, epoch)
                writer.add_scalar('val/adversarial', v_metrics['j_adv'].avg, epoch)
        elif cfg.loss.type == 'entropic':
            writer.add_scalar('loss/entropic', t_metrics['j_e'].avg, epoch)
            writer.add_scalar('val/entropic', v_metrics['j_e'].avg, epoch)
            if cfg.adv.who != 'no_adv':
                writer.add_scalar('loss/adversarial', t_metrics['j_adv'].avg, epoch)
                writer.add_scalar('val/adversarial', v_metrics['j_adv'].avg, epoch)
        elif cfg.loss.type == 'softmax':
            writer.add_scalar('loss/softmax', t_metrics['j_s'].avg, epoch)
            writer.add_scalar('val/softmax', v_metrics['j_s'].avg, epoch)
        writer.add_scalar('val/auc', v_metrics['auc'].avg, epoch)
        writer.add_scalar('val/conf', v_metrics['conf'].avg, epoch)

        #  training information on console
        val_time = time.time() - train_time - epoch_time  # validation+metrics writer+save model time
        info = 'ep:{} train:{} val:{} t:{:.1f}s v:{:.1f}s'.format(
            epoch, dict(t_metrics), dict(v_metrics), train_time, val_time)
        logger.info(info)

        # save best model and current model
        f_name = '{}_curr.pth'.format(cfg.exp_name)  # current model
        if curr_score > best_score:
            best_score = curr_score
            f_name = '{}_best.pth'.format(cfg.exp_name)  # best model
            logger.info('Saving best model at epoch: {}'.format(epoch))
        save_checkpoint(f_name, model, epoch, opt, best_score, scheduler=scheduler)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info('early stop')
                break

    logger.info('Training finished')

# ========================== Evaluation ========================== #
    test_file = data_dir/cfg.data.test_file
    test_ds = ImagenetDataset(test_file, cfg.data.imagenet_path, val_tf)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True
    )
    logger.info('Evaluating in validation and test datasets')
    arrays_path = '{}_curr_val_arr.npz'.format(cfg.exp_name)  # Current model is the model from last epoch
    save_eval_arrays(model, val_loader, device, cfg.batch_size, arrays_path)
    arrays_path = '{}_curr_test_arr.npz'.format(cfg.exp_name)
    save_eval_arrays(model, test_loader, device, cfg.batch_size, arrays_path)
    # Evaluate best model if exists
    best_path = Path('{}_best.pth'.format(cfg.exp_name))
    if best_path.exists():
        load_checkpoint(model, best_path, opt=None, device=device)
        arrays_path = '{}_best_val_arr.npz'.format(cfg.exp_name)
        save_eval_arrays(model, val_loader, device, cfg.batch_size, arrays_path)
        arrays_path = '{}_best_test_arr.npz'.format(cfg.exp_name)
        save_eval_arrays(model, test_loader, device, cfg.batch_size, arrays_path)


if __name__ == '__main__':
    main()
