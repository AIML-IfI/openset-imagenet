""" Training script for Open-set Classification on Imagenet"""
import random
import time
from sys import stderr
from pathlib import Path
import os
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as tf
from vast.tools import set_device_gpu
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from loguru import logger
import metrics
from dataset import ImagenetDataset
from model import ResNet50
from losses import AverageMeter, EarlyStopping, EntropicLoss, ObjectoLoss
from adversary import add_random_noise, add_gaussian_noise, fgsm_attack, decay_epsilon


# Global objects:
BEST_SCORE = 0.0    # Best validation score
START_EPOCH = 0     # Initial training epoch


def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    """ Saves a training checkpoint.

    Args:
        f_name(str): File name.
        model(torch module): Pytorch model.
        epoch(int): Current epoch.
        opt(torch optimiser): Current optimiser.
        best_score_(float): Current best score.
        scheduler(torch lr_scheduler): Pytorch scheduler.
    """
    # If model is DistributedDataParallel extracts the module.
    if isinstance(model, DistributedDataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    data = {"epoch": epoch + 1,
            "model_state_dict": state,
            "opt_state_dict": opt.state_dict(),
            "best_score": best_score_}
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, f_name)


def check_config(cfg):
    """ Placeholder to perform parameter checks.

    Args:
        cfg: Configuration file
    """
    if (cfg.adv.who != "no_adv") and cfg.dist.distributed:
        message = "Test Message"
        raise Exception(message)


def load_checkpoint(model, checkpoint, opt=None, device="cpu", scheduler=None):
    """ Loads a checkpoint, if the model was saved using DistributedDataParallel, removes the word
    'module' from the state_dictionary keys to load it in a single device. If fine-tuning model then
    optimizer should be none to start from clean optimizer parameters.

    Args:
        model (torch nn.module): Requires a model to load the state dictionary.
        checkpoint (Path): File path.
        opt (torch optimiser): An optimiser to load the state dictionary. Defaults to None.
        device (str): Device to load the checkpoint. Defaults to 'cpu'.
        scheduler (torch lr_scheduler): Learning rate scheduler. Defaults to None.
    """
    file_path = Path(checkpoint)
    if file_path.is_file():  # First check if file exists
        checkpoint = torch.load(file_path, map_location=device)
        key = list(checkpoint["model_state_dict"].keys())[0]
        # If module was saved as DistributedDataParallel then removes the world "module"
        # from dictionary keys
        if key[:6] == "module":
            new_state_dict = OrderedDict()
            for k_i, v_i in checkpoint["model_state_dict"].items():
                key = k_i[7:]  # remove "module"
                new_state_dict[key] = v_i
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if opt is not None:  # Load optimizer state
            opt.load_state_dict(checkpoint["opt_state_dict"])

        if scheduler is not None:  # Load scheduler state
            scheduler.load_state_dict(checkpoint["scheduler"])

        del checkpoint
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        return start_epoch, best_score
    else:
        raise Exception("Checkpoint file not found")


def predict(scores, threshold):
    """ Returns the class and max score of the sample. If the max score<threshold returns -1.

    Args:
        scores(tensor): Softmax scores of all classes and samples in batch.
        threshold(float): Minimum score to classify as a known sample.

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
        tuple: Tuple that has in fist position a tensor with indices of correctly predicted samples.
    """
    with torch.no_grad():
        scores = torch.nn.functional.softmax(logits, dim=1)
        pred = predict(scores, threshold)
        correct = (targets >= 0) * (pred[:, 0] == targets)
        return torch.nonzero(correct, as_tuple=True)


def train(model, data_loader, optimiser, device, loss_fn, trackers, cfg):
    """ Main training loop.

    Args:
        model (torch.model): Model
        data_loader (torch.DataLoader): DataLoader
        optimiser (torch optimiser): Optimiser
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
    for images, labels in data_loader:
        model.train()  # To collect batch-norm statistics
        batch_len = labels.shape[0]  # Samples in current batch
        optimiser.zero_grad(set_to_none=True)
        images = images.to(device)
        labels = labels.to(device)

        # If the gradient with respect to the input is needed
        if cfg.adv.who == "fgsm":
            images.requires_grad_()
            images.grad = None

        # Forward pass
        logits, features = model(images)

        # Calculate loss
        if cfg.loss.type == "objectosphere":
            j = loss_fn(features, logits, labels, cfg.loss.alpha)
            trackers["j_o"].update(loss_fn.objecto_value, batch_len)
            trackers["j_e"].update(loss_fn.entropic_value, batch_len)
        elif cfg.loss.type in ["BGsoftmax", "entropic", "softmax"]:
            j = loss_fn(logits, labels)
            trackers["j"].update(j.item(), batch_len)
        # Backward pass
        j.backward()

        if cfg.adv.who == "no_adv":  # If training without adversarial samples
            optimiser.step()
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
            if cfg.adv.mode == "filter":
                correct_idx = filter_correct(logits=logits, targets=labels, threshold=cfg.threshold)
                num_adv_samples = len(correct_idx[0])
            elif cfg.adv.mode == "full":  # Perturb all samples
                correct_idx = torch.arange(batch_len, requires_grad=False, device=device)
                num_adv_samples = len(correct_idx)
            trackers["num_adv"].update(num_adv_samples)

            # Create perturbed samples
            if num_adv_samples > 0:
                correct_im = images[correct_idx]
                if cfg.adv.who == "gaussian":
                    adv_im, adv_label = add_gaussian_noise(
                        image=correct_im,
                        loc=0,
                        std=cfg.adv.std,
                        device=device)
                elif cfg.adv.who == "random":
                    adv_im, adv_label = add_random_noise(
                        image=correct_im,
                        epsilon=cfg.adv.epsilon,
                        device=device)
                elif cfg.adv.who == "fgsm":
                    correct_im_grad = images.grad[correct_idx]
                    adv_im, adv_label = fgsm_attack(
                        image=correct_im,
                        epsilon=cfg.adv.epsilon,
                        grad=correct_im_grad,
                        device=device)

                # forward pass with adversarial samples
                logits, features = model(adv_im)
                j_a = None
                if cfg.loss.type == "objectosphere":
                    j_a = loss_fn(features, logits, adv_label, cfg.loss.alpha)
                elif cfg.loss.type == "entropic":
                    j_a = loss_fn(logits, adv_label)
                trackers["j_adv"].update(j_a.item(), num_adv_samples)
                j_a.backward()
            optimiser.step()


def validate(model, data_loader, device, loss_fn, n_classes, trackers, cfg):
    """ Validation loop.
    Args:
        model (torch.model): Model
        data_loader (torch dataloader): DataLoader
        device (cuda): cuda id
        loss_fn: Loss function
        n_classes(int): Total number of classes
        trackers(dict): Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset all validation metrics
    for metric in trackers.values():
        metric.reset()

    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)  # size of dataset
        all_targets = torch.empty(data_len, device=device, dtype=torch.int64).detach()
        all_scores = torch.empty((data_len, n_classes), device=device).detach()

        for i, (images, labels) in enumerate(data_loader):
            batch_len = labels.shape[0]  # current batch size, last batch has different value
            images = images.to(device)
            labels = labels.to(device)
            logits, features = model(images)
            scores = torch.nn.functional.softmax(logits, dim=1)

            if cfg.loss.type == "objectosphere":
                j = loss_fn(features, logits, labels, cfg.loss.alpha)
                trackers["j_o"].update(loss_fn.objecto_value, batch_len)
                trackers["j_e"].update(loss_fn.entropic_value, batch_len)
            elif cfg.loss.type in ["BGsoftmax", "entropic", "softmax"]:
                j = loss_fn(logits, labels)
                trackers["j"].update(j.item(), batch_len)

            # accumulate partial results in empty tensors
            start_ix = i * cfg.batch_size
            all_targets[start_ix: start_ix + batch_len] = labels
            all_scores[start_ix: start_ix + batch_len] = scores

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
        if cfg.loss.type == "softmax":
            min_unk_score = 0.0
            auc = metrics.auc_score_multiclass(all_targets, all_scores)

        elif cfg.loss.type in ["entropic", "objectosphere"]:
            min_unk_score = 1 / n_classes
            # max_kn_scores = torch.max(all_scores, dim=1)[0]
            auc = metrics.auc_score_binary(all_targets, all_scores)

        elif cfg.loss.type == "BGsoftmax":
            min_unk_score = 0.0
            # Removes last column of scores to use only known classes
            all_scores = all_scores[:, :-1]

            # Replaces the biggest class label with -1
            biggest_label = data_loader.dataset.unique_classes[-1]
            all_targets[all_targets == biggest_label] = -1

            auc = metrics.auc_score_binary(all_targets, all_scores)

        kn_conf, kn_count, neg_conf, neg_count = metrics.confidence(
            scores=all_scores,
            target_labels=all_targets,
            offset=min_unk_score)
        trackers["auc"].update(auc, data_len)
        if kn_count:
            trackers["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
            trackers["conf_unk"].update(neg_conf, neg_count)


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main function wrapped in hydra.main who does the setup and manages the config file.

    Args:
        cfg (DictConfig): Configuration file.
    """
    # Setting the logger

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    check_config(cfg)

    if cfg.dist.distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cfg.dist.port
        #print(f"Distributed training at: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        #print(f"Using {cfg.dist.gpus} gpus")
        mp.spawn(worker, nprocs=cfg.dist.gpus, args=(cfg,out_dir, ))
    else:
        gpu = 0
        worker(gpu, cfg, out_dir,)
        # print(Path(hydra.utils.get_original_cwd()))
        # print(HydraConfig.get().runtime.output_dir)


def worker(gpu, cfg, out_dir,):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        gpu(int): GPU index.
        cfg(DictConfig): Configuration dictionary.
        out_dir(Path): Output directory
    """
    # referencing best score and setting seeds
    global BEST_SCORE
    global START_EPOCH
    set_seeds(cfg.seed)

    if cfg.dist.distributed:
        # initialize process group. For rank needs to be the global rank among all the processes
        rank = gpu
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=cfg.dist.gpus,
            rank=rank)
    else:
        rank = 0  # Only one rank

    # Configure logger. Log only on first process. Validate only on first process.
    if rank == 0:
        msg_format = "{time:DD_MM_HH:mm} {message}"
        logger.configure(handlers=[{"sink": stderr, "level": "INFO", "format": msg_format}])
        logger.add(
            sink= out_dir / cfg.log_name,
            format=msg_format,
            level="INFO",
            mode='w')

    # Set image transformations
    train_tr = tf.Compose(
        [tf.Resize(256),
         tf.RandomCrop(224),
         tf.RandomHorizontalFlip(0.5),
         tf.ToTensor()])

    val_tr = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])

    # create datasets
    train_file = Path(cfg.data.train_file)
    val_file = Path(cfg.data.val_file)

    if train_file.exists() and val_file.exists():
        train_ds = ImagenetDataset(
            csv_file=train_file,
            imagenet_path=cfg.data.imagenet_path,
            transformation=train_tr)
        val_ds = ImagenetDataset(
            csv_file=val_file,
            imagenet_path=cfg.data.imagenet_path,
            transformation=val_tr)

        # If using garbage class, replaces label -1 to maximum label + 1
        if cfg.loss.type == "BGsoftmax":
            # Only change the unknown label of the training dataset
            train_ds.replace_negative_label()
            val_ds.replace_negative_label()
    else:
        raise FileNotFoundError("train/validation file does not exist")

    # Create data loader
    if cfg.dist.distributed:
        train_sampler = DistributedSampler(train_ds, seed=cfg.seed, drop_last=False)
        #val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,)
        #sampler = val_sampler)

    # setup device
    device = torch.device(f"cuda:{gpu}")
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
    if cfg.loss.type == "objectosphere":
        loss = ObjectoLoss(n_classes, cfg.loss.w, cfg.loss.xi)
    elif cfg.loss.type == "entropic":
        loss = EntropicLoss(n_classes, cfg.loss.w)
    elif cfg.loss.type == "softmax":
        loss = torch.nn.CrossEntropyLoss().to(device)
    elif cfg.loss.type == "BGsoftmax":
        class_weights = train_ds.calculate_class_weights().to(device)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)
    model.to(device)

    # Create optimizer
    if cfg.opt.type == "sgd":
        opt = torch.optim.SGD(params=model.parameters(), lr=cfg.opt.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(params=model.parameters(), lr=cfg.opt.lr)

    # Learning rate scheduler
    if cfg.opt.decay > 0:
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg.opt.decay,
            gamma=cfg.opt.gamma,
            verbose=True)
    else:
        scheduler = None

    # Resume a training from a checkpoint
    if cfg.checkpoint is not None:
        # Get the relative path of the checkpoint wrt train.py
        if cfg.train_mode == "finetune": # TODO: Simplify the modes, finetune is not necessary
            START_EPOCH, _ = load_checkpoint(
                model=model,
                checkpoint=cfg.checkpoint,
                opt=None,
                device=device,
                scheduler=None)
            BEST_SCORE = 0
        else:
            START_EPOCH, BEST_SCORE = load_checkpoint(
                model=model,
                checkpoint=cfg.checkpoint,
                opt=opt,
                device=device,
                scheduler=scheduler)
        if rank == 0:
            logger.info(f"Best score of loaded model: {BEST_SCORE:.3f}. 0 is for fine tuning")
            logger.info(f"Loaded {cfg.checkpoint} at epoch {START_EPOCH}")

    # wrap model in ddp
    if cfg.dist.distributed:
        model = DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)
    # Needed to keep initial epsilon when using decay
    start_epsilon = cfg.adv.epsilon
    adv_who = cfg.adv.who  # Store initial adversary
    # Checks if train the model without adversarial samples for a number of epochs then add
    # adversarial samples.
    if cfg.adv.wait > 0:
        cfg.adv.who = "no_adv"

    # Set the final epoch
    cfg.epochs = START_EPOCH + cfg.epochs

    # Print info to console and setup summary writer

    writer = None
    if rank == 0:
        # Info on console
        logger.info("============ Data ============")
        logger.info(f"train_len:{len(train_ds)}, labels:{train_ds.label_count}")
        logger.info(f"val_len:{len(val_ds)}, labels:{val_ds.label_count}")
        logger.info(f"Total batch size: {cfg.batch_size * cfg.dist.gpus}")
        logger.info("========== Training ==========")
        logger.info(f"Initial epoch: {START_EPOCH}")
        logger.info(f"Last epoch: {cfg.epochs}")
        logger.info(f"General mode: {cfg.train_mode}")
        logger.info(f"Batch size: {cfg.batch_size}")
        logger.info(f"workers: {cfg.workers}")
        logger.info(f"Adversary: {cfg.adv.who}")
        logger.info(f"Adversary mode: {cfg.adv.mode}")
        logger.info(f"Loss: {cfg.loss.type}")
        logger.info(f"Optimiser: {cfg.opt.type}")
        logger.info(f"Learning rate: {cfg.opt.lr}")
        logger.info(f"Device: {device}")
        logger.info("Training...")
        writer = SummaryWriter(log_dir=out_dir)

    for epoch in range(START_EPOCH, cfg.epochs):
        epoch_time = time.time()

        if cfg.dist.distributed:
            train_sampler.set_epoch(epoch)

        if (cfg.adv.wait > 0) and (epoch >= cfg.adv.wait):
            cfg.adv.who = adv_who

        # calculate epsilon
        if (cfg.adv.who in ["fgsm", "random"]) and 0 < cfg.adv.mu < 1 and cfg.adv.decay > 0:
            cfg.adv.epsilon = decay_epsilon(
                start_eps=start_epsilon,
                mu=cfg.adv.mu,
                curr_epoch=epoch,
                wait_epochs=cfg.adv.decay
            )
            logger.info(f"epsilon:{cfg.adv.epsilon:.4f}")

        # training loop
        train(
            model=model,
            data_loader=train_loader,
            optimiser=opt,
            device=device,
            loss_fn=loss,
            trackers=t_metrics,
            cfg=cfg)

        train_time = time.time() - epoch_time

        # validation loop
        validate(
            model=model,
            data_loader=val_loader,
            device=device,
            loss_fn=loss,
            n_classes=n_classes,
            trackers=v_metrics,
            cfg=cfg)

        curr_score = v_metrics["auc"].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        if rank == 0:
            # Logging metrics to tensorboard object
            if cfg.loss.type == "objectosphere":
                writer.add_scalar("train/objecto", t_metrics["j_o"].avg, epoch)
                writer.add_scalar("train/entropic", t_metrics["j_e"].avg, epoch)
                writer.add_scalar("val/objecto", v_metrics["j_o"].avg, epoch)
                writer.add_scalar("val/entropic", v_metrics["j_e"].avg, epoch)
            elif cfg.loss.type in ["entropic", 'softmax', 'BGsoftmax']:
                writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
                writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
            if cfg.adv.who != "no_adv":
                writer.add_scalar("train/adversarial", t_metrics["j_adv"].avg, epoch)
                writer.add_scalar("val/adversarial", v_metrics["j_adv"].avg, epoch)
            # Validation metrics
            writer.add_scalar("val/auc", v_metrics["auc"].avg, epoch)
            writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
            writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

            #  training information on console
            # validation+metrics writer+save model time
            val_time = time.time() - train_time - epoch_time
            logger.info(
                f"ep:{epoch} "
                f"train:{dict(t_metrics)} "
                f"val:{dict(v_metrics)} "
                f"t:{train_time:.1f}s "
                f"v:{val_time:.1f}s")

            # save best model and current model
            ckpt_name = str(out_dir / cfg.name) + "_curr.pth"
            if curr_score > BEST_SCORE:
                BEST_SCORE = curr_score
                ckpt_name = str(out_dir / cfg.name) + "_best.pth"
                # ckpt_name = f"{cfg.name}_best.pth"  # best model
                logger.info(f"Saving best model at epoch: {epoch}")
            save_checkpoint(ckpt_name, model, epoch, opt, BEST_SCORE, scheduler=scheduler)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info("early stop")
                break

    logger.info("Training finished")


if __name__ == "__main__":
    main()
