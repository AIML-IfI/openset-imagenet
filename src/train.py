import random
import time
import torch
import os
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
import metrics
from adversary import add_random_noise, add_gaussian_noise, fgsm_attack, decay_epsilon
from dataset import ImagenetDataset
from model import ResNet50
from losses import AverageMeter, EarlyStopping, EntropicLoss, ObjectoLoss
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.optim import lr_scheduler
from vast.tools import set_device_gpu
# Distributed training
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from sys import stderr
import hydra
from omegaconf import DictConfig
from os import getcwd

# Global objects:
best_score = 0.0    # Best validation score
start_epoch = 0     # Initial training epoch


def set_seeds(seed):
    """ Sets the seed for different sources of randomness

    Args:
        seed (int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    """ Saves a training checkpoint.

    Args:
        f_name: File name.
        model: Pytorch model.
        epoch: Current epoch.
        opt: Current optimizer.
        best_score_: Current best score.
        scheduler: Pytorch scheduler.
    """
    # If model is DistributedDataParallel extracts the module.
    state = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()

    data = {'epoch': epoch + 1,
            'model_state_dict': state,
            'opt_state_dict': opt.state_dict(),
            'best_score': best_score_}
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()
    torch.save(data, f_name)


def check_config(cfg):
    """ Placeholder to perform parameter checks.

    Args:
        cfg: Configuration file
    """
    if (cfg.adv.who != 'no_adv') and cfg.dist.distributed:
        m = "Test Message"
        raise Exception(m)


def load_checkpoint(model, checkpoint, opt=None, device='cpu', scheduler=None):
    """ Loads a checkpoint, if the model was saved using DistributedDataParallel, removes the word
    'module' from the state_dictionary keys to load it in a single device. If fine-tuning model then
    optimizer should be none to start from clean optimizer parameters.

    Args:
        model (torch nn.module): Requires a model to load the state dictionary.
        checkpoint (Path): File path.
        opt (torch optimizer, optional): An optimiser to load the state dictionary. Defaults to None.
        device (str, optional): Device to load the checkpoint. Defaults to 'cpu'.
        scheduler (torch lr_scheduler, optional): Learning rate scheduler. Defaults to None.
    """
    global best_score
    global start_epoch

    file_path = Path(checkpoint)
    if file_path.is_file:  # First check if file exists
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

        if opt is not None:  # Load optimizer state
            opt.load_state_dict(checkpoint['opt_state_dict'])

        start_epoch = checkpoint['epoch']

        if scheduler is not None:  # Load scheduler state
            scheduler.load_state_dict(checkpoint['scheduler'])

        if 'best_score' in checkpoint:  # Load best score
            best_score = checkpoint['best_score']
            logger.info('best score of loaded model: {:.3f}'.format(checkpoint['best_score']))
        logger.info('loaded {} at epoch {}'.format(file_path, checkpoint['epoch']))
        del checkpoint
    else:
        logger.info('Checkpoint file not found')


def predict(scores, threshold):
    """ Returns the class and max score of the sample. If the max score<threshold returns -1.

    Args:
        scores: Softmax scores of all classes and samples in batch.
        threshold: Minimum score to classify as a known sample.

    Returns:
        Tensor with predicted class and score.
    """
    pred_score, pred_class = torch.max(scores, dim=1)
    unk = pred_score < threshold
    pred_class[unk] = -1
    return torch.stack((pred_class, pred_score), dim=1)


def filter_correct(logits, targets, threshold):
    """Returns the indices of correctly predicted known samples.

    Args:
        logits (tensor): Logits tensor
        targets (tensor): Targets tensor
        threshold (float): Minimum score for the target to be classified as known.

    Returns:
        tuple: Tuple containing in fist position the tensor with indices of correctly predicted samples.
    """
    with torch.no_grad():
        scores = torch.nn.functional.softmax(logits, dim=1)
        pred = predict(scores, threshold)
        correct = (targets >= 0) * (pred[:, 0] == targets)
        return torch.nonzero(correct, as_tuple=True)


def train(model, data_loader, optimizer, device, loss_fn, trackers, cfg):
    """ Main training loop.

    Args:
        model (torch.model): Model
        data_loader (torch.DataLoader): DataLoader
        optimizer (torch.Optimizer): Optimizer
        device (cuda): cuda id
        loss_fn: Loss function
        trackers: Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset dictionary of training metrics
    for metric in trackers.values():
        metric.reset()

    j = None

    # training loop
    for x, t in data_loader:
        model.train()  # To collect batch normalisation statistics
        batch_len = t.shape[0]  # Samples in current batch
        optimizer.zero_grad(set_to_none=True)
        x = x.to(device)
        t = t.to(device)

        # If the gradient with respect to the input is needed
        if cfg.adv.who == 'fgsm':
            x.requires_grad_()
            x.grad = None

        # Forward pass
        logits, features = model(x)

        # Calculate loss
        if cfg.loss.type == 'objectosphere':
            j = loss_fn(features, logits, t, cfg.loss.alpha)
            trackers['j_o'].update(loss_fn.objecto_value, batch_len)
            trackers['j_e'].update(loss_fn.entropic_value, batch_len)
        elif cfg.loss.type in ['BGsoftmax', 'entropic', 'softmax']:
            j = loss_fn(logits, t)
            trackers['j'].update(j.item(), batch_len)
        # Backward pass
        j.backward()

        if cfg.adv.who == 'no_adv':  # If training without adversarial samples
            optimizer.step()
        else:
            # Steps:
            #   Select samples to perturb
            #   Create adv samples
            #   Calculate adv loss
            #   Backward pass
            model.eval()  # To stop batch normalisation statistics

            # Get the candidates to adversarial samples
            num_adv_samples = 0
            correct_idx = None
            # Perturb corrected classified samples
            if cfg.adv.mode == 'filter':
                correct_idx = filter_correct(logits=logits, targets=t, threshold=cfg.threshold)
                num_adv_samples = len(correct_idx[0])
            elif cfg.adv.mode == 'full':  # Perturb all samples
                correct_idx = torch.arange(batch_len, requires_grad=False, device=device)
                num_adv_samples = len(correct_idx)
            trackers['num_adv'].update(num_adv_samples)

            # Create perturbed samples
            if num_adv_samples > 0:
                x_corr = x[correct_idx]
                if cfg.adv.who == 'gaussian':
                    x_adv, t_adv = add_gaussian_noise(x=x_corr, loc=0, std=cfg.adv.std, device=device)
                elif cfg.adv.who == 'random':
                    x_adv, t_adv = add_random_noise(x=x_corr, epsilon=cfg.adv.epsilon, device=device)
                elif cfg.adv.who == 'fgsm':
                    x_corr_grad = x.grad[correct_idx]
                    x_adv, t_adv = fgsm_attack(x=x_corr, epsilon=cfg.adv.epsilon, grad=x_corr_grad, device=device)

                # forward pass with adversarial samples
                logits, features = model(x_adv)
                j_a = None
                if cfg.loss.type == 'objectosphere':
                    j_a = loss_fn(features, logits, t_adv, cfg.loss.alpha)
                elif cfg.loss.type == 'entropic':
                    j_a = loss_fn(logits, t_adv)
                trackers['j_adv'].update(j_a.item(), num_adv_samples)
                j_a.backward()
            optimizer.step()


def validate(model, loader, device, loss_fn, n_classes, trackers, cfg):
    """

    Args:
        model:
        loader:
        device:
        loss_fn:
        n_classes:
        trackers:
        cfg:

    Returns:

    """
    for metric in trackers.values():
        metric.reset()

    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)  # size of dataset
        all_targets = torch.empty(data_len, device=device, dtype=torch.int64).detach()
        all_scores = torch.empty((data_len, n_classes), device=device).detach()

        for i, (x, t) in enumerate(loader):
            batch_len = t.shape[0]  # current batch size, last batch has different value
            x = x.to(device)
            t = t.to(device)
            logits, features = model(x)
            scores = torch.nn.functional.softmax(logits, dim=1)

            if cfg.loss.type == 'objectosphere':
                j = loss_fn(features, logits, t, cfg.loss.alpha)
                trackers['j_o'].update(loss_fn.objecto_value, batch_len)
                trackers['j_e'].update(loss_fn.entropic_value, batch_len)
            elif cfg.loss.type in ['entropic', 'softmax', 'BGsoftmax']:
                j = loss_fn(logits, t)
                trackers['j'].update(j.item(), batch_len)

            # accumulate partial results in empty tensors
            ix = i * cfg.batch_size
            all_targets[ix: ix + batch_len] = t
            all_scores[ix: ix + batch_len] = scores

        # Validation cases for different losses:
        # Softmax:  metric: multiclass AUC
        #           score, target: without unknowns
        # Entropic: metric: Binary AUC (kn vs unk)
        #           score: does not have unknown class
        #           target: unknown class -1
        # BGsoftmax: metric: Binary AUC (kn vs unk)
        #           score: has additional class for unknown samples, remove it
        #           target: unknown class -1
        min_unk_score = None
        if cfg.loss.type == 'softmax':
            min_unk_score = 0.0
            auc = metrics.auc_score_multiclass(all_targets, all_scores)

        elif cfg.loss.type in ['entropic', 'objectosphere']:
            min_unk_score = 1 / n_classes
            # max_kn_scores = torch.max(all_scores, dim=1)[0]
            auc = metrics.auc_score_binary(all_targets, all_scores)

        elif cfg.loss.type == 'BGsoftmax':
            min_unk_score = 0.0
            all_scores = all_scores[:, :-1]  # Removes last column of scores to use only known classes

            # Replaces the biggest class label with -1
            biggest_label = loader.dataset.unique_classes[-1]
            all_targets[all_targets == biggest_label] = -1

            auc = metrics.auc_score_binary(all_targets, all_scores)

        kn_conf, kn_count, neg_conf, neg_count = metrics.confidence(all_scores, all_targets, min_unk_score)
        trackers['auc'].update(auc, data_len)
        if kn_count:
            trackers['conf_kn'].update(kn_conf, kn_count)
        if neg_count:
            trackers['conf_unk'].update(neg_conf, neg_count)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Setting the logger
    msg_format = "{time:DD_MM_HH:mm} {message}"
    logger.configure(
        handlers=[{"sink": stderr, "level": "INFO", "format": msg_format}])
    logger.add(cfg.log_name, format=msg_format, level="INFO", mode='w')
    
    check_config(cfg)
    
    if cfg.dist.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = cfg.dist.port
        print('\nDistributed training at: {}:{}'.format(
            os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))
        print('Using {} gpus'.format(cfg.dist.gpus))
        mp.spawn(worker, nprocs=cfg.dist.gpus, args=(cfg,))
    else:
        gpu = 0
        worker(gpu, cfg)


# noinspection PyArgumentList
def worker(gpu, cfg):
    """
    Args:
        gpu: 
        cfg: 
    Returns:
    """
    # referencing best score and setting seeds
    global best_score
    set_seeds(cfg.seed)
    
    if cfg.dist.distributed:
        # initialize process group. For rank needs to be the global rank among all the processes
        rank = gpu
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=cfg.dist.gpus,
            rank=rank
        )
    else:
        rank = 0  # Only one rank
        logger.info('Training is using 1 gpu')

    # Set image transformations
    train_tr = tf.Compose(
        [tf.Resize(256),
         tf.RandomCrop(224),
         tf.RandomHorizontalFlip(0.5),
         tf.ToTensor()]
    )
    
    val_tr = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()]
    )

    # create datasets
    data_dir = Path(cfg.data.data_dir)
    train_file = data_dir / cfg.data.train_file
    val_file = data_dir / cfg.data.val_file

    if train_file.exists() and val_file.exists():
        train_ds = ImagenetDataset(csv_file=train_file, imagenet_path=cfg.data.imagenet_path, transformation=train_tr)
        val_ds = ImagenetDataset(csv_file=val_file, imagenet_path=cfg.data.imagenet_path, transformation=val_tr)

        # If using garbage class, replaces label -1 to maximum label + 1
        if cfg.loss.type == 'BGsoftmax':
            # Only change the unknown label of the training dataset
            train_ds.replace_negative_label()
            val_ds.replace_negative_label()
    else:
        raise FileNotFoundError('train/validation file does not exist')

    # Create data loader
    if cfg.dist.distributed:
        sampler = DistributedSampler(train_ds, seed=cfg.seed, drop_last=False)
    else:
        sampler = None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True
    )

    # setup device
    device = torch.device('cuda:{}'.format(gpu))
    set_device_gpu(index=gpu)

    # Callbacks
    early_stopping = None
    if cfg.patience > 0:
        early_stopping = EarlyStopping(patience=cfg.patience)

    # Set dictionaries to keep track of the losses
    t_metrics = defaultdict(AverageMeter)
    v_metrics = defaultdict(AverageMeter)

    # set loss
    loss = None
    if train_ds.has_negatives():
        # number of classes - 1 when training with unknowns
        n_classes = train_ds.label_count - 1
    else:
        n_classes = train_ds.label_count
    if cfg.loss.type == 'objectosphere':
        loss = ObjectoLoss(n_classes, cfg.loss.w, cfg.loss.xi)
    elif cfg.loss.type == 'entropic':
        loss = EntropicLoss(n_classes, cfg.loss.w)
    elif cfg.loss.type == 'softmax':
        loss = torch.nn.CrossEntropyLoss().to(device)
    elif cfg.loss.type == 'BGsoftmax':
        class_weights = train_ds.calculate_class_weights().to(device)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)
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
        checkpoint_path = original_wd / cfg.checkpoint
        if cfg.train_mode == 'finetune':
            load_checkpoint(model, checkpoint_path, None, device, None)
            logger.info('Fine-tuning: Setting best score to 0')
            best_score = 0
        else:
            load_checkpoint(model, checkpoint_path, opt, device, scheduler)

    # wrap model in ddp
    if cfg.dist.distributed:
        model = DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)
    # Needed to keep initial epsilon when using decay
    start_epsilon = cfg.adv.epsilon
    adv_who = cfg.adv.who  # Store initial adversary
    # Checks if train the model without adversarial for a nr. of epochs then add adversarial samples.
    if cfg.adv.wait > 0:
        cfg.adv.who = 'no_adv'

    # Set the final epoch
    global start_epoch
    cfg.epochs = start_epoch + cfg.epochs

    # Print info to console and setup summary writer
    writer = None
    if rank == 0:
        # Info on console
        logger.info('========== Data ==========')
        logger.info('train_len:{}, labels:{}'.format(len(train_ds), train_ds.label_count))
        logger.info('val_len:{}, labels:{}'.format(len(val_ds), val_ds.label_count))
        logger.info('Total batch size: {}'.format(cfg.batch_size * cfg.dist.gpus))
        logger.info('========== Training ==========')
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
        writer = SummaryWriter(log_dir=getcwd())

    for epoch in range(start_epoch, cfg.epochs):
        epoch_time = time.time()

        if cfg.dist.distributed:
            sampler.set_epoch(epoch)

        if (cfg.adv.wait > 0) and (epoch >= cfg.adv.wait):
            cfg.adv.who = adv_who

        # calculate epsilon
        if (cfg.adv.who in ['fgsm', 'random']) and 0 < cfg.adv.mu < 1 and cfg.adv.decay > 0:
            cfg.adv.epsilon = decay_epsilon(
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

        if rank == 0:
            # Logging metrics to tensorboard object
            if cfg.loss.type == 'objectosphere':
                writer.add_scalar('train/objecto', t_metrics['j_o'].avg, epoch)
                writer.add_scalar('train/entropic', t_metrics['j_e'].avg, epoch)
                writer.add_scalar('val/objecto', v_metrics['j_o'].avg, epoch)
                writer.add_scalar('val/entropic', v_metrics['j_e'].avg, epoch)
            elif cfg.loss.type in ['entropic', 'softmax', 'BGsoftmax']:
                writer.add_scalar('train/loss', t_metrics['j'].avg, epoch)
                writer.add_scalar('val/loss', v_metrics['j'].avg, epoch)
            if cfg.adv.who != 'no_adv':
                writer.add_scalar('train/adversarial', t_metrics['j_adv'].avg, epoch)
                writer.add_scalar('val/adversarial', v_metrics['j_adv'].avg, epoch)
            # Validation metrics
            writer.add_scalar('val/auc', v_metrics['auc'].avg, epoch)
            writer.add_scalar('val/conf_kn', v_metrics['conf_kn'].avg, epoch)
            writer.add_scalar('val/conf_unk', v_metrics['conf_unk'].avg, epoch)

            #  training information on console
            # validation+metrics writer+save model time
            val_time = time.time() - train_time - epoch_time
            info = 'ep:{} train:{} val:{} t:{:.1f}s v:{:.1f}s'.format(
                epoch, dict(t_metrics), dict(v_metrics), train_time, val_time
            )
            logger.info(info)

            # save best model and current model
            f_name = '{}_curr.pth'.format(cfg.exp_name)  # current model
            if curr_score > best_score:
                best_score = curr_score
                f_name = '{}_best.pth'.format(cfg.exp_name)  # best model
                logger.info('Saving best model at epoch: {}'.format(epoch))
            # if epoch % 10 == 0:
            #     f_name = '{}_curr_ep{}.pth'.format(cfg.exp_name, epoch)
            save_checkpoint(f_name, model, epoch, opt, best_score, scheduler=scheduler)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info('early stop')
                break

    logger.info('Training finished')


if __name__ == '__main__':
    main()
