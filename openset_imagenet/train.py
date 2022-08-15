import random
import time
import sys
import pathlib
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from vast.tools import set_device_gpu, set_device_cpu, device
from loguru import logger
from .metrics import confidence, auc_score_binary, auc_score_multiclass, predict_objectosphere
from .dataset import ImagenetDataset
from .model import ResNet50
from .losses import AverageMeter, EarlyStopping, EntropicLoss, ObjectoLoss
import tqdm


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
        opt(torch optimizer): Current optimizer.
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


def load_checkpoint(model, checkpoint, opt=None, device="cpu", scheduler=None):
    """ Loads a checkpoint, if the model was saved using DistributedDataParallel, removes the word
    'module' from the state_dictionary keys to load it in a single device. If fine-tuning model then
    optimizer should be none to start from clean optimizer parameters.

    Args:
        model (torch nn.module): Requires a model to load the state dictionary.
        checkpoint (Path): File path.
        opt (torch optimizer): An optimizer to load the state dictionary. Defaults to None.
        device (str): Device to load the checkpoint. Defaults to 'cpu'.
        scheduler (torch lr_scheduler): Learning rate scheduler. Defaults to None.
    """
    file_path = pathlib.Path(checkpoint)
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

        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        return start_epoch, best_score
    else:
        raise Exception(f"Checkpoint file '{checkpoint}' not found")


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


def train(model, data_loader, optimizer, loss_fn, trackers, cfg):
    """ Main training loop.

    Args:
        model (torch.model): Model
        data_loader (torch.DataLoader): DataLoader
        optimizer (torch optimizer): optimizer
        loss_fn: Loss function
        trackers: Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset dictionary of training metrics
    for metric in trackers.values():
        metric.reset()

    j = None

    # training loop
    if not cfg.parallel:
        data_loader = tqdm.tqdm(data_loader)
    for images, labels in data_loader:
        model.train()  # To collect batch-norm statistics
        batch_len = labels.shape[0]  # Samples in current batch
        optimizer.zero_grad()
        images = device(images)
        labels = device(labels)

        # Forward pass
        logits, features = model(images)

        # Calculate loss
        if cfg.loss.type == "objectosphere":
            j = loss_fn(features, logits, labels, cfg.loss.alpha)
            trackers["j_o"].update(loss_fn.objecto_value, batch_len)
            trackers["j_e"].update(loss_fn.entropic_value, batch_len)
        elif cfg.loss.type in ["garbage", "entropic", "softmax"]:
            j = loss_fn(logits, labels)
            trackers["j"].update(j.item(), batch_len)
        # Backward pass
        j.backward()
        optimizer.step()


def validate(model, data_loader, loss_fn, n_classes, trackers, cfg):
    """ Validation loop.
    Args:
        model (torch.model): Model
        data_loader (torch dataloader): DataLoader
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
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, n_classes), requires_grad=False))

        for i, (images, labels) in enumerate(data_loader):
            batch_len = labels.shape[0]  # current batch size, last batch has different value
            images = device(images)
            labels = device(labels)
            logits, features = model(images)
            scores = torch.nn.functional.softmax(logits, dim=1)

            if cfg.loss.type == "objectosphere":
                j = loss_fn(features, logits, labels, cfg.loss.alpha)
                trackers["j_o"].update(loss_fn.objecto_value, batch_len)
                trackers["j_e"].update(loss_fn.entropic_value, batch_len)
            elif cfg.loss.type in ["garbage", "entropic", "softmax"]:
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
        # garbage: metric: Binary AUC (kn vs unk)
        #           score: has additional class for unknown samples, remove it
        #           target: unknown class -1
        min_unk_score = None
        if cfg.loss.type == "softmax":
            min_unk_score = 0.0
            auc = auc_score_multiclass(all_targets, all_scores)

        elif cfg.loss.type in ["entropic", "objectosphere"]:
            min_unk_score = 1 / n_classes
            # max_kn_scores = torch.max(all_scores, dim=1)[0]
            auc = auc_score_binary(all_targets, all_scores)

        elif cfg.loss.type == "garbage":
            min_unk_score = 0.0
            # Removes last column of scores to use only known classes
            all_scores = all_scores[:, :-1]

            # Replaces the biggest class label with -1
            biggest_label = data_loader.dataset.unique_classes[-1]
            all_targets[all_targets == biggest_label] = -1

            auc = auc_score_binary(all_targets, all_scores)

        kn_conf, kn_count, neg_conf, neg_count = confidence(
            scores=all_scores,
            target_labels=all_targets,
            offset=min_unk_score)
        trackers["auc"].update(auc, data_len)
        if kn_count:
            trackers["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
            trackers["conf_unk"].update(neg_conf, neg_count)



def get_arrays(model, loader):
    """ Extract deep features, logits and targets for all dataset. Returns numpy arrays

    Args:
        model (torch model): Model.
        loader (torch dataloader): Data loader.
    """
    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        logits_dim = model.logits.out_features  # logits output classes
        features_dim = model.logits.in_features  # features dimensionality
        all_targets = torch.empty(data_len, device="cpu")  # store all targets
        all_logits = torch.empty((data_len, logits_dim), device="cpu")   # store all logits
        all_feat = torch.empty((data_len, features_dim), device="cpu")   # store all features
        all_scores = torch.empty((data_len, logits_dim), device="cpu")

        index = 0
        for images, labels in tqdm.tqdm(loader):
            curr_b_size = labels.shape[0]  # current batch size, very last batch has different value
            images = device(images)
            labels = device(labels)
            logit, feature = model(images)
            score = torch.nn.functional.softmax(logit, dim=1)
            # accumulate results in all_tensor
            all_targets[index:index + curr_b_size] = labels.detach().cpu()
            all_logits[index:index + curr_b_size] = logit.detach().cpu()
            all_feat[index:index + curr_b_size] = feature.detach().cpu()
            all_scores[index:index + curr_b_size] = score.detach().cpu()
            index += curr_b_size
        return(
            all_targets.numpy(),
            all_logits.numpy(),
            all_feat.numpy(),
            all_scores.numpy())


def worker(cfg):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        cfg (NameSpace): Configuration of the experiment
    """
    # referencing best score and setting seeds
    set_seeds(cfg.seed)

    BEST_SCORE = 0.0    # Best validation score
    START_EPOCH = 0     # Initial training epoch

    # Configure logger. Log only on first process. Validate only on first process.
#    msg_format = "{time:DD_MM_HH:mm} {message}"
    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink= cfg.output_directory / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')

    # Set image transformations
    train_tr = transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(0.5),
         transforms.ToTensor()])

    val_tr = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    # create datasets
    train_file = pathlib.Path(cfg.data.train_file.format(cfg.protocol))
    val_file = pathlib.Path(cfg.data.val_file.format(cfg.protocol))

    if train_file.exists() and val_file.exists():
        train_ds = ImagenetDataset(
            csv_file=train_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=train_tr
        )
        val_ds = ImagenetDataset(
            csv_file=val_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=val_tr
        )

        # If using garbage class, replaces label -1 to maximum label + 1
        if cfg.loss.type == "garbage":
            # Only change the unknown label of the training dataset
            train_ds.replace_negative_label()
            val_ds.replace_negative_label()
        elif cfg.loss.type == "softmax":
            # Remove all unknown labels
            train_ds.remove_negative_label()
            val_ds.remove_negative_label()

    else:
        raise FileNotFoundError("train/validation file does not exist")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,)

    # setup device
    if cfg.gpu is not None:
        set_device_gpu(index=cfg.gpu)
    else:
        logger.warning("No GPU device selected, training will be extremely slow")
        set_device_cpu()

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
        loss = torch.nn.CrossEntropyLoss()
    elif cfg.loss.type == "garbage":
        class_weights = device(train_ds.calculate_class_weights())
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)

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
                scheduler=None)
            BEST_SCORE = 0
        else:
            START_EPOCH, BEST_SCORE = load_checkpoint(
                model=model,
                checkpoint=cfg.checkpoint,
                opt=opt,
                scheduler=scheduler)
        logger.info(f"Best score of loaded model: {BEST_SCORE:.3f}. 0 is for fine tuning")
        logger.info(f"Loaded {cfg.checkpoint} at epoch {START_EPOCH}")

    device(model)

    # Set the final epoch
    cfg.epochs = START_EPOCH + cfg.epochs

    # Print info to console and setup summary writer

    # Info on console
    logger.info("============ Data ============")
    logger.info(f"train_len:{len(train_ds)}, labels:{train_ds.label_count}")
    logger.info(f"val_len:{len(val_ds)}, labels:{val_ds.label_count}")
    logger.info("========== Training ==========")
    logger.info(f"Initial epoch: {START_EPOCH}")
    logger.info(f"Last epoch: {cfg.epochs}")
    logger.info(f"General mode: {cfg.train_mode}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"workers: {cfg.workers}")
    logger.info(f"Loss: {cfg.loss.type}")
    logger.info(f"optimizer: {cfg.opt.type}")
    logger.info(f"Learning rate: {cfg.opt.lr}")
    logger.info(f"Device: {cfg.gpu}")
    logger.info("Training...")
    writer = SummaryWriter(log_dir=cfg.output_directory, filename_suffix="-"+cfg.log_name)

    for epoch in range(START_EPOCH, cfg.epochs):
        epoch_time = time.time()

        # training loop
        train(
            model=model,
            data_loader=train_loader,
            optimizer=opt,
            loss_fn=loss,
            trackers=t_metrics,
            cfg=cfg)

        train_time = time.time() - epoch_time

        # validation loop
        validate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss,
            n_classes=n_classes,
            trackers=v_metrics,
            cfg=cfg)

        curr_score = v_metrics["auc"].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        # Logging metrics to tensorboard object
        if cfg.loss.type == "objectosphere":
            writer.add_scalar("train/objecto", t_metrics["j_o"].avg, epoch)
            writer.add_scalar("train/entropic", t_metrics["j_e"].avg, epoch)
            writer.add_scalar("val/objecto", v_metrics["j_o"].avg, epoch)
            writer.add_scalar("val/entropic", v_metrics["j_e"].avg, epoch)
        elif cfg.loss.type in ["entropic", 'softmax', 'garbage']:
            writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
            writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
        # Validation metrics
        writer.add_scalar("val/auc", v_metrics["auc"].avg, epoch)
        writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
        writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

        #  training information on console
        # validation+metrics writer+save model time
        val_time = time.time() - train_time - epoch_time
        def pretty_print(d):
            #return ",".join(f'{k}: {float(v):1.3f}' for k,v in dict(d).items())
            return dict(d)

        logger.info(
            f"ep:{epoch} "
            f"train:{pretty_print(t_metrics)} "
            f"val:{pretty_print(v_metrics)} "
            f"t:{train_time:.1f}s "
            f"v:{val_time:.1f}s")

        # save best model and current model
        ckpt_name = str(cfg.output_directory / cfg.name) + "_curr.pth"
        if curr_score > BEST_SCORE:
            BEST_SCORE = curr_score
            ckpt_name = str(cfg.output_directory / cfg.name) + "_best.pth"
            # ckpt_name = f"{cfg.name}_best.pth"  # best model
            logger.info(f"Saving best model at epoch: {epoch}")
        save_checkpoint(ckpt_name, model, epoch, opt, BEST_SCORE, scheduler=scheduler)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info("early stop")
                break

    # clean everything
    del model
    torch.cuda.empty_cache()
    logger.info("Training finished")
