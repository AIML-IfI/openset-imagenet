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
import vast
from loguru import logger
from .metrics import confidence, auc_score_binary, auc_score_multiclass
from .dataset import ImagenetDataset
from .model import ResNet50, ResNet50Proser
from .losses import AverageMeter, EarlyStopping, EntropicOpensetLoss
import tqdm
from .model import set_seeds, save_checkpoint, load_checkpoint




def train_proser(model, data_loader, optimizer, loss_fn, trackers, cfg):
    """ Main training loop.

    Args:
        model (ResNet50Proser): Model
        data_loader (torch.DataLoader): DataLoader
        optimizer (torch optimizer): optimizer
        loss_fn: Loss function
        trackers: Dictionary of trackers
        cfg: General configuration structure
    """
    alpha = cfg.algorithm.alpha
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)

    # Reset dictionary of training metrics
    for metric in trackers.values():
        metric.reset()

    mixed_class_label = data_loader.dataset.label_count

    # training loop
    if not cfg.parallel:
        data_loader = tqdm.tqdm(data_loader)

    #print('in the train function, class label count: ', mixed_class_label  )

    # MG comment: I am not sure if this is a good idea.
    # I do not know how batchnorm handles forward to be called several times before backward is called
    model.train()

    for images, labels in data_loader:
        batch_len = labels.shape[0]
        optimizer.zero_grad()
        images = device(images)
        labels = device(labels)

        # split the batch into two parts.
        # the first part is mixed and assigned to unknown
        # the second contains the original samples
        mixed_count = batch_len//2
        clean_count = batch_len - mixed_count
        indexes = device(torch.randperm(mixed_count))

        mixed_labels = labels[:mixed_count]
        clean_labels = labels[mixed_count:]


        ##### FIRST PART: mixtures
        # create mixtures by extracting features from the first half of the batch
        middle_layer_features = model.first_blocks(images[:mixed_count])


        # mix some them
        beta = beta_distribution.sample([]).item()
        mixed_middle_layer_features = beta * middle_layer_features + (1-beta) * middle_layer_features[indexes]


        # Masking the pairs of the same class,
        # passing only mixed embeddings created from different classes.
        mixed_middle_layer_features = mixed_middle_layer_features[mixed_labels != mixed_labels[indexes]]
        #print('in train fun, mixed_middle_layer_features shape: ', mixed_middle_layer_features.shape)
        if len(mixed_middle_layer_features):
            # forward these to the second part of the network
            mixed_logits, mixed_dummy_score, _ = model.last_blocks(mixed_middle_layer_features)
            mixed_logits = torch.cat((mixed_logits, mixed_dummy_score[:,None]), dim=1)

            # train the mixed logits to be of the unknown class
            loss1 = loss_fn(mixed_logits, device(torch.as_tensor([mixed_class_label] * len(mixed_logits))))
        else:
            loss1 = device(torch.zeros(1))


        ##### SECOND PART: clean data
        # get network output for second half of the batch
        clean_logits, clean_dummy_score, _ = model(images[mixed_count:])
        clean_logits = torch.cat((clean_logits, clean_dummy_score[:,None]), dim=1)

        # Setting the known class logit to -1e9 to force a classifier placeholder to be the second highest logit.
        modified_clean_logits = clean_logits.clone()
        for i in range(clean_count):
            modified_clean_logits[i][clean_labels[i]] = -1e9
        # quicker variant:?
        # modified_clean_logits[clean_labels] = -1e9


        ##### THIRD PART: losses

        # train the clean outputs with the clean labels
        loss2 = loss_fn(clean_logits, clean_labels)

        # (eq. 5 in the paper) 2nd term in l1, forcing a placeholder classifier to be the 2nd closest to the known sample.
        loss3 = loss_fn(modified_clean_logits, device(torch.as_tensor([mixed_class_label] * clean_count)))

        # the final loss is a weighted combination of the former
        j = cfg.algorithm.lambda0 * loss1 + cfg.algorithm.lambda1 * loss2 + cfg.algorithm.lambda2 * loss3

        # track results
        trackers["j"].update(j.item(), batch_len)
        trackers["j1"].update(loss1.item(), batch_len)
        trackers["j2"].update(loss2.item(), batch_len)
        trackers["j3"].update(loss3.item(), batch_len)

        # Backward pass
        j.backward()
        optimizer.step()


def compute_bias(model, data_loader):
    """Bias computation to tune the dummy (background class) logit magnitude"""
    biases = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = device(images)
            labels = device(labels)
            logits, dummy_score, _ = model(images)
            biases.append((torch.max(logits, dim=1) - dummy_score).cpu())

    # Returns bias corresponding to the 5th percentile element, so that 95% of the data is classified as knowns.
    biases = torch.sort(torch.cat(biases))
    return biases[int(len(biases)*0.05)]


# We will use our own default validation process, instead of taking PROSER's approach
def validate_proser(model, data_loader, loss_fn, trackers, cfg):
    """ Validation loop.
    Args:
        model (torch.model): Model
        data_loader (torch dataloader): DataLoader
        loss_fn: Loss function
        trackers(dict): Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset all validation metrics
    for metric in trackers.values():
        metric.reset()

    # since we are at the validation dataset, we have C+1 classes.
    # Hence, out model is supposed to have C scores, plus the dummy score, which we use for evaluation
    n_classes = data_loader.dataset.label_count

    # for PROSER, we always have one additional output
    min_unk_score = 0.
    unknown_class = n_classes - 1
    last_valid_class = -1

    model.eval()
    with torch.no_grad():
        data_len = len(data_loader.dataset)  # size of dataset
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, n_classes), requires_grad=False))

        for i, (images, labels) in enumerate(data_loader):
            batch_len = labels.shape[0]  # current batch size, last batch has different value
            images = device(images)
            labels = device(labels)
            logits, dummy, features = model(images)

            # compute probabilities for all classes (excluding unknown)
            mixed_logits = torch.cat((logits, dummy[:,None]), dim=1)
            scores = torch.nn.functional.softmax(mixed_logits, dim=1)

            # store all scores and all targets
            start_ix = i * cfg.batch_size
            all_targets[start_ix: start_ix + batch_len] = labels
            all_scores[start_ix: start_ix + batch_len] = scores

        kn_conf, kn_count, neg_conf, neg_count = confidence(
            scores=all_scores,
            target_labels=all_targets,
            offset=min_unk_score,
            unknown_class = unknown_class,
            last_valid_class = last_valid_class)
        if kn_count:
            trackers["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
            trackers["conf_unk"].update(neg_conf, neg_count)


def get_arrays(model, loader, pretty=False):
    """ Extract deep features, logits and targets for all dataset. Returns numpy arrays

    Args:
        model (torch model): Model.
        loader (torch dataloader): Data loader.
    """

    n_classes = loader.dataset.label_count - (2 if -2 in loader.dataset.unique_classes else 1)
    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        features_dim = model.resnet_base.logits.in_features # features dimensionality
        #print("logits dim: ", logits_dim, "feature dim:", features_dim)

        all_targets = torch.empty(data_len, device="cpu")  # store all targets
        all_logits = torch.empty((data_len, n_classes), device="cpu")   # store all logits
        all_feat = torch.empty((data_len, features_dim), device="cpu")   # store all features
        all_scores = torch.empty((data_len, n_classes), device="cpu")

        index = 0
        if pretty:
            loader = tqdm.tqdm(loader)
        for images, labels in loader:
            curr_b_size = labels.shape[0]  # current batch size, very last batch has different value
            images = device(images)
            labels = device(labels)
            logits, dummy, feature = model(images)
            # Compute softmax over all logits, including the dummy class
            scores = torch.nn.functional.softmax(torch.cat((logits, dummy.view(logits.shape[0], 1)), dim=1), dim=1)
            # We always subtract the scores for the unknown class, after computing softmax
            scores = scores[:,:-1]
            # accumulate results in all_tensor
            all_targets[index:index + curr_b_size] = labels.detach().cpu()
            all_logits[index:index + curr_b_size] = logits.detach().cpu()
            all_feat[index:index + curr_b_size] = feature.detach().cpu()
            all_scores[index:index + curr_b_size] = scores.detach().cpu()
            index += curr_b_size
        return(
            all_targets.numpy(),
            all_logits.numpy(),
            all_feat.numpy(),
            all_scores.numpy())


def worker(cfg, dummy_count_index=0):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        cfg (NameSpace): Configuration of the experiment
    """
    # referencing best score and setting seeds
    set_seeds(cfg.seed)
    cfg.algorithm.dummy_count = cfg.algorithm.dummy_counts[dummy_count_index]

    BEST_SCORE = 0.0    # Best validation score
    START_EPOCH = 0     # Initial training epoch
    cfg.log_name = cfg.log_name.format(cfg.loss.type, cfg.algorithm.type, cfg.epochs, cfg.algorithm.dummy_count)
    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink= pathlib.Path(cfg.output_directory) / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')
    logger.info('file created')

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

        # remove the negative label from PROSER training set
        train_ds.remove_negative_label()
        # for the val set, we change them to be in class C+1
        val_ds.replace_negative_label()

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
    # number of classes when training with extra garbage class for unknowns, or when unknowns are removed
    n_classes = train_ds.label_count + (1 if cfg.loss.type == "garbage" else 0)

    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Create the model
    base = ResNet50(
        fc_layer_dim=n_classes,
        out_features=n_classes,
        logit_bias=False
    )
    model = ResNet50Proser(
        dummy_count = cfg.algorithm.dummy_count,
        fc_layer_dim=n_classes,
        resnet_base = base,
        loss_type=cfg.loss.type
    )

    device(model)

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
        START_EPOCH, BEST_SCORE = load_checkpoint(
            model=model,
            checkpoint=cfg.checkpoint,
            opt=opt,
            scheduler=scheduler)
        logger.info(f"Best score of loaded model: {BEST_SCORE:.3f}. 0 is for fine tuning")
        logger.info(f"Loaded {cfg.checkpoint} at epoch {START_EPOCH}")
    else:
        # for PROSER, we always start with a pre-trained model, which we need to load here
        start_epoch, best_score = load_checkpoint(
            model = base,
            checkpoint = cfg.model_path.format(cfg.output_directory, cfg.loss.type, "threshold", "curr"),
            opt = None,
            scheduler = None
        )

        logger.info(f"Loaded {cfg.model_path.format(cfg.output_directory, cfg.loss.type, 'threshold', 'curr')} and taking model from epoch {start_epoch} that achieved best score {best_score}")

    # Print info to console and setup summary writer
    logger.info("============ Data ============")
    logger.info(f"train_len:{len(train_ds)}, labels:{train_ds.label_count}")
    logger.info(f"val_len:{len(val_ds)}, labels:{val_ds.label_count}")
    logger.info(f"dummy_count: {cfg.algorithm.dummy_count}")
    logger.info("========== Training ==========")
    logger.info(f"Initial epoch: {START_EPOCH}")
    logger.info(f"Last epoch: {cfg.epochs}")
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
        train_proser(
            model=model,
            data_loader=train_loader,
            optimizer=opt,
            loss_fn=loss,
            trackers=t_metrics,
            cfg=cfg)

        train_time = time.time() - epoch_time

        # validation loop
        validate_proser(
            model=model,
            data_loader=val_loader,
            loss_fn=loss,
            trackers=v_metrics,
            cfg=cfg)

        curr_score = v_metrics["conf_kn"].avg + v_metrics["conf_unk"].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        # Logging metrics to tensorboard object
        writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
        # Validation metrics
        writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
        writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

        #  training information on console
        # validation+metrics writer+save model time
        val_time = time.time() - train_time - epoch_time
        def pretty_print(d):
            #return ",".join(f'{k}: {float(v):1.3f}' for k,v in dict(d).items())
            return dict(d)

        logger.info(
            f"loss:{cfg.loss.type} "
            f"protocol:{cfg.protocol} "
            f"ep:{epoch} "
            f"train:{pretty_print(t_metrics)} "
            f"val:{pretty_print(v_metrics)} "
            f"t:{train_time:.1f}s "
            f"v:{val_time:.1f}s")

        # save best model and current model
        ckpt_name = cfg.algorithm.output_model_path.format(cfg.output_directory, cfg.loss.type, cfg.algorithm.type, cfg.epochs, cfg.algorithm.dummy_count, "curr")
        save_checkpoint(ckpt_name, model, epoch, opt, curr_score, scheduler=scheduler)

        if curr_score > BEST_SCORE:
            BEST_SCORE = curr_score
            ckpt_name = cfg.algorithm.output_model_path.format(cfg.output_directory, cfg.loss.type, cfg.algorithm.type, cfg.epochs, cfg.algorithm.dummy_count, "best")
            # ckpt_name = f"{cfg.name}_best.pth"  # best model
            logger.info(f"Saving best model {ckpt_name} at epoch: {epoch}")
            save_checkpoint(ckpt_name, model, epoch, opt, BEST_SCORE, scheduler=scheduler)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info("early stop")
                break

    # clean everything
    del model
    logger.info("Training finished")
