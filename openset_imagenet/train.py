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
from .model import ResNet50
from .losses import AverageMeter, EarlyStopping, EntropicOpensetLoss
import tqdm
from .context import approaches, architectures, data_prep, tools
import openset_imagenet
import pandas as pd
import pickle
from vast import opensetAlgos


def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = Falsegg



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


def load_checkpoint(model, checkpoint, opt=None, scheduler=None):
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
#        breakpoint()
        checkpoint = torch.load(file_path, map_location=vast.tools._device)
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

    if cfg.loss.type == "garbage":
        min_unk_score = 0.
        unknown_class = n_classes - 1
        last_valid_class = -1
    else:
        min_unk_score = 1. / n_classes
        unknown_class = -1
        last_valid_class = None


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

            j = loss_fn(logits, labels)
            trackers["j"].update(j.item(), batch_len)

            # accumulate partial results in empty tensors
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



def compose_dicts(targets, features, logits):
    df_dim = features.shape[-1] if len(features.shape) > 1 else 1
    df_data = pd.DataFrame(torch.hstack((targets, features, logits)).numpy(), columns=[
        'gt'] + [f'feat_{i+1}' for i in range(df_dim)] + [f'log_{j+1}' for j in range(logits.shape[-1])])
    df_data['gt'] = df_data['gt'].astype(np.int32)

    df_group = df_data.groupby('gt')
    feat_dict = (df_group.apply(lambda x: list(
    map(list, zip(*[x[f'feat_{i+1}'] for i in range(df_dim)])))).to_dict())
    for k in feat_dict:
        feat_dict[k] = torch.Tensor(feat_dict[k])
    logit_dict = (df_group.apply(lambda x: list(
        map(list, zip(*[x[f'log_{i+1}'] for i in range(logits.shape[-1])])))).to_dict())
    for k in logit_dict:
        logit_dict[k] = torch.Tensor(logit_dict[k])

    count_feat, count_logits = 0, 0

    for k in feat_dict:
        count_feat += feat_dict[k].shape[0]
        count_logits += logit_dict[k].shape[0]

    logger.debug('\n')
    logger.info(f'Number of samples included in the dict: {count_feat}')
    logger.info(f'Number of classes (i.e. # dict keys): {len(list(feat_dict.keys()))}')
    return feat_dict, logit_dict

def postprocess_train_data(targets, features, logits):
    # Note: OpenMax uses only the training samples that get correctly classified by the
          # underlying, extracting DNN to train its model.logger.debug('\n')
    #print(f'{... post-processing:')

    with torch.no_grad():
        # OpenMax only uses KKCs for training
        known_idxs = (targets >= 0).squeeze()

        targets_kkc, features_kkc, logits_kkc = targets[
            known_idxs], features[known_idxs], logits[known_idxs]

        class_predicted = torch.max(logits_kkc, axis=1).indices
        correct_idxs = targets_kkc.squeeze() == class_predicted

        logger.info(
            f'Correct classifications: {torch.sum(correct_idxs).item()}')
        logger.info(
            f'Incorrect classifications: {torch.sum(~correct_idxs).item()}')
        logger.info(
            f'Number of samples after post-processing: {targets_kkc[correct_idxs].shape[0]}')
        logger.info(
            f'Number of unique classes after post-processing: {len(collect_pos_classes(targets_kkc[correct_idxs]))}')

        return targets_kkc[correct_idxs], features_kkc[correct_idxs], logits_kkc[correct_idxs]

def collect_pos_classes(targets):
    targets_unique = torch.unique(targets, sorted=True)
    pos_classes = targets_unique[targets_unique >=
                                  0].numpy().astype(np.int32).tolist()
    return pos_classes


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

def save_models(all_hyper_param_models,pos_classes, cfg):
        # integrating returned models in a data structure as required by <self.approach>_Inference()
        hparam_combo_to_model = defaultdict(list)

        for i in range(len(all_hyper_param_models)):
            hparam_combo_to_model[all_hyper_param_models[i][0]].append(all_hyper_param_models[i][1])

        logger.info(f'Trained models associated with hyperparameters: {list(hparam_combo_to_model.keys())}')
        for key in hparam_combo_to_model:
            hparam_combo_to_model[key] = dict(hparam_combo_to_model[key])

            # store models per hyperparameter combination as a (hparam_combo, model)-tuple
            model_name = f'p{cfg.protocol}_traincls({"+".join(cfg.train_classes)})_{cfg.alg.lower()}_{key}_{cfg.hyp.distance_metric}_dnn_{cfg.loss.type}.pkl'

            file_handler = open(cfg.output_directory / model_name, 'wb')
            
            #obj_serializable = {'approach_train': cfg.alg, 'model_name': model_name, 
            #        'hparam_combo': key, 'distance_metric': cfg.distance_metric, 'instance': {'protocol': cfg.protocol, 'gpu': cfg.gpu,
            #            'ku_target': cfg.known_unknown_target, 'uu_target': cfg.unknown_unknown_target, 'model_path': cfg.output_directory, 'log_path': self.log_path,
            #            'oscr_path': self.oscr_path, 'train_cls': self.train_classes, 'architecture': self.architecture, 'used_dnn': self.used_dnn, 'fpr_thresholds': self.fpr_thresholds}, 'model':  hparam_combo_to_model[key]}

            obj_serializable = {'approach_train': cfg.alg, 'model_name': model_name, 
                    'hparam_combo': key, 'distance_metric': cfg.hyp.distance_metric, 'instance': {'protocol': cfg.protocol, 'gpu': cfg.gpu,
                        'ku_target': cfg.known_unknown_target, 'uu_target': cfg.unknown_unknown_target, 'model_path': cfg.output_directory}, 'model':  hparam_combo_to_model[key]}
            
                    
            pickle.dump(obj_serializable, file_handler)

            """
            Important: Since the <approach>_Inference() function in the vast package sorts the 
            keys of the collated model, the semantic of the returned probabilities depends on 
            the type of the dictionary keys. For example, when sorting is applied on the 'stringified'
            integer classes, the column indices of the returned probabilities tensor do not necessarily
            correspond with the integer class targets. Hence, the assertion for integer type below. 
            """
            assert sum([isinstance(k, int) for k in hparam_combo_to_model[key].keys()]) == len(
                list(hparam_combo_to_model[key].keys())), 'dictionarys keys are not of type "int"'

        """
        SANITY CHECKS
        """
        assert len(set([el[0] for el in all_hyper_param_models])) == len(
            hparam_combo_to_model.keys()), 'missing entries for hyperparameter combinations'
        assert [(el == len(pos_classes)) for el in [len(hparam_combo_to_model[k].keys())
                                                    for k in hparam_combo_to_model.keys()]], 'model misses training class(es)'


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
            # remove the negative label from softmax training set, not from val set!
            train_ds.remove_negative_label()


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

    #vast openmax uses device.index which can be achiavable through this assignment. 
    dev = torch.device(cfg.gpu if torch.cuda.is_available() else 'cpu')
    
    # Callbacks
    early_stopping = None
    if cfg.patience > 0:
        early_stopping = EarlyStopping(patience=cfg.patience)

    # Set dictionaries to keep track of the losses
    t_metrics = defaultdict(AverageMeter)
    v_metrics = defaultdict(AverageMeter)

    # set loss
    loss = None
    if cfg.loss.type == "entropic":
        # number of classes - 1 since we have no label for unknown
        n_classes = train_ds.label_count - 1
    else:
        # number of classes when training with extra garbage class for unknowns, or when unknowns are removed
        n_classes = train_ds.label_count

    if cfg.loss.type == "entropic":
        # We select entropic loss using the unknown class weights from the config file
        loss = EntropicOpensetLoss(n_classes, cfg.loss.w)
    elif cfg.loss.type == "softmax":
        # We need to ignore the index only for validation loss computation
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif cfg.loss.type == "garbage":
        # We use balanced class weights
        class_weights = device(train_ds.calculate_class_weights())
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)
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

    if cfg.alg in ["OpenMax", "EVM"]:
        print(f"Let me try {cfg.alg}")
        #dnn_dict = torch.load(cfg.dnn_features)
        logger.debug('\n')
        #logger.info(f'Info on loaded feature-extracting model: {dnn_dict["model_name"]}, best trainings performance {dnn_dict["eval_metric_opt"]:.6f} ({dnn_dict["eval_metric"]}) achieved in epoch {dnn_dict["epoch_opt"]}')
        #for k, v in dnn_dict.items():
        #    print(k)
        
        #pretrained_dnn = getattr(architectures, dnn_dict['instance']['architecture'])(
        #        feature_dim=dnn_dict['instance']['df_dim'], num_classes=dnn_dict['instance']['num_cls'])
        
        #pretrained_dnn = getattr(architectures, 'ResNet50_Feature')(feature_dim=512, num_classes=116 )
        #pretrained_dnn.load_state_dict(
        #        dnn_dict['model_state_dict'], strict=False)
        print(cfg.gpu, cfg.alg)
        
        #extractor = getattr(data_prep, f'{cfg.alg}Extractor')(pretrained_dnn, cfg.gpu, cfg.alg)


        #train_feat, pos_classes = extractor.extract_train_features(train_loader)

            # create model
        #n_classes = train_ds.label_count
        suffix = cfg.suffix
        #model = openset_imagenet.ResNet50(fc_layer_dim=n_classes, out_features=n_classes, logit_bias=False)
        print(cfg.output_directory)
        start_epoch, best_score = openset_imagenet.train.load_checkpoint(model, pathlib.Path(cfg.output_directory / (str(cfg.loss.type)+suffix+".pth")))
        print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
        device(model)

        if cfg.alg == 'OpenMax':
            alg_hyperparameters=[cfg.hyp.tailsize, cfg.hyp.distance_multiplier, cfg.hyp.translateAmount, cfg.hyp.distance_metric, cfg.hyp.alpha]
        elif cfg.alg == 'EVM':
            alg_hyperparameters = [cfg.evm_parameters.tailsize, cfg.evm_parameters.cover_threshold,cfg.evm_parameters.distance_multiplier, cfg.evm_parameters.distance_metric, cfg.evm_parameters.chunk_size]
        
        print(f'{(cfg.alg).lower()}_hyperparams')
        hyperparams = getattr(approaches, f'{(cfg.alg).lower()}_hyperparams')(*alg_hyperparameters)


        print("========== Training  ==========")

        print("Feature extraction on training data:")
       
       # extracting arrays for training data
        gt, logits, features, scores = get_arrays(
                model=model,
                loader=train_loader
                )
        gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)

        kkc = collect_pos_classes(gt)

        targets, features, logits = postprocess_train_data(gt, features, logits)
        #print(kkc)
        pos_classes = collect_pos_classes(targets)

        feat_dict, _ = compose_dicts(targets, features, logits)
        
        #approach = getattr(approaches, cfg.alg)
        
        #print(approach)

        logger.debug('\n')
        logger.info(f'Starting {cfg.alg} Training Procedure:')

        training_fct = getattr(opensetAlgos, f'{cfg.alg}_Training')
        
        #performs training on all parameter combinations
        #Training method returns iterator over (hparam_combo, (class, {model}))
        all_hyper_param_models = list(training_fct(
            pos_classes_to_process=pos_classes, features_all_classes=feat_dict, args=hyperparams, gpu=dev.index, models=None))
        
        save_models(all_hyper_param_models, pos_classes, cfg)


        exit()

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

        curr_score = v_metrics["conf_kn"].avg + v_metrics["conf_unk"].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        # Logging metrics to tensorboard object
        writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
        writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
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
        ckpt_name = str(cfg.output_directory / cfg.name) + "_curr.pth"
        save_checkpoint(ckpt_name, model, epoch, opt, curr_score, scheduler=scheduler)

        if curr_score > BEST_SCORE:
            BEST_SCORE = curr_score
            ckpt_name = str(cfg.output_directory / cfg.name) + "_best.pth"
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
