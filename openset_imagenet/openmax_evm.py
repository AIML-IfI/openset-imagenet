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
#from .context import approaches, architectures, data_prep, tools
import openset_imagenet
import pandas as pd
import pickle
from vast import opensetAlgos
from .train import set_seeds, save_checkpoint, load_checkpoint, get_arrays
import argparse
from collections import namedtuple





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




def save_models(all_hyper_param_models,pos_classes, cfg):
        # integrating returned models in a data structure as required by <self.approach>_Inference()
        hparam_combo_to_model = defaultdict(list)

        for i in range(len(all_hyper_param_models)):
            hparam_combo_to_model[all_hyper_param_models[i][0]].append(all_hyper_param_models[i][1])

        logger.info(f'Trained models associated with hyperparameters: {list(hparam_combo_to_model.keys())}')
        for key in hparam_combo_to_model:
            hparam_combo_to_model[key] = dict(hparam_combo_to_model[key])

            # store models per hyperparameter combination as a (hparam_combo, model)-tuple
            #model_name = f'p{cfg.protocol}_traincls({"+".join(cfg.train_classes)})_{cfg.alg.lower()}_{key}_{cfg.hyp.distance_metric}_dnn_{cfg.loss.type}.pkl'
            model_name = f'{cfg.loss.type}_{cfg.algorithm.type}_{key}_{cfg.algorithm.distance_metric}.pkl'

            file_handler = open(cfg.output_directory / model_name, 'wb')
            
            #obj_serializable = {'approach_train': cfg.alg, 'model_name': model_name, 
            #        'hparam_combo': key, 'distance_metric': cfg.distance_metric, 'instance': {'protocol': cfg.protocol, 'gpu': cfg.gpu,
            #            'ku_target': cfg.known_unknown_target, 'uu_target': cfg.unknown_unknown_target, 'model_path': cfg.output_directory, 'log_path': self.log_path,
            #            'oscr_path': self.oscr_path, 'train_cls': self.train_classes, 'architecture': self.architecture, 'used_dnn': self.used_dnn, 'fpr_thresholds': self.fpr_thresholds}, 'model':  hparam_combo_to_model[key]}

            obj_serializable = {'approach_train': cfg.algorithm.type, 'model_name': model_name, 
                    'hparam_combo': key, 'distance_metric': cfg.algorithm.distance_metric, 'instance': {'protocol': cfg.protocol, 'gpu': cfg.gpu,
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

def openmax_hyperparams(tailsize, dist_mult, translate_amount, dist_metric, alpha):

    hparam_string = f'--tailsize {" ".join(str(e) for e in tailsize)} --distance_multiplier {" ".join(str(e) for e in dist_mult)} --distance_metric {dist_metric}'

    parser = argparse.ArgumentParser()
    parser, _ = getattr(opensetAlgos, 'OpenMax_Params')(parser)

    ns_obj = parser.parse_args(hparam_string.split())
    ns_obj.alpha = alpha
    return ns_obj

def evm_hyperparams(tailsize, cover_thres, dist_mult, dist_metric, chunk):
    hyperparams = namedtuple('EVM_Hyperparams', [
        'tailsize', 'cover_threshold', 'distance_multiplier', 'distance_metric', 'chunk_size'])
    return hyperparams(tailsize, cover_thres, dist_mult, dist_metric, chunk)

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

    # create datasets
    train_file = pathlib.Path(cfg.data.train_file.format(cfg.protocol))
  

    if train_file.exists() :
        train_ds = ImagenetDataset(
            csv_file=train_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=train_tr
        )
        train_ds.remove_negative_label()
    else:
        raise FileNotFoundError("train file does not exist")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True)


    # setup device
    if cfg.gpu is not None:
        set_device_gpu(index=cfg.gpu)
    else:
        logger.warning("No GPU device selected, training will be extremely slow")
        set_device_cpu()

    #vast openmax uses device.index which can be achiavable through this assignment. 
    dev = torch.device(cfg.gpu if torch.cuda.is_available() else 'cpu')
    
    n_classes = train_ds.label_count 
    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)
    device(model)


    print(f"Let me try {cfg.algorithm.type}")
    logger.debug('\n')

    print("GPU:", cfg.gpu, cfg.algorithm.type)

    suffix = cfg.suffix

    print(cfg.output_directory)
    #cfg.algorithm.base_model.format(cfg.protocol)

    #start_epoch, best_score = load_checkpoint(model, pathlib.Path(cfg.output_directory / (str(cfg.loss.type)+suffix+".pth")))
    start_epoch, best_score = load_checkpoint(model, checkpoint=cfg.algorithm.base_model.format(cfg.protocol) )

    print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)

    if cfg.algorithm.type== 'openmax':
        alg_hyperparameters=[cfg.algorithm.tailsize, cfg.algorithm.distance_multiplier, cfg.algorithm.translateAmount, cfg.algorithm.distance_metric, cfg.algorithm.alpha_om]
        hyperparams = openmax_hyperparams(*alg_hyperparameters)
    elif cfg.algorithm.type == 'evm':
        alg_hyperparameters = [cfg.algorithm.tailsize, cfg.algorithm.cover_threshold,cfg.algorithm.distance_multiplier, cfg.algorithm.distance_metric, cfg.algorithm.chunk_size]
        hyperparams = evm_hyperparams(*alg_hyperparameters)
    
    print(f'{(cfg.algorithm.type)}_hyperparams finished')
    

    #hyperparams = getattr(approaches, f'{cfg.alg}_hyperparams')(*alg_hyperparameters)


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
    logger.info(f'Starting {cfg.algorithm.type} Training Procedure:')

    training_fct = getattr(opensetAlgos, f'{vars(cfg.alg_dict)[cfg.algorithm.type]}_Training')
    
    #performs training on all parameter combinations
    #Training method returns iterator over (hparam_combo, (class, {model}))
    all_hyper_param_models = list(training_fct(
        pos_classes_to_process=pos_classes, features_all_classes=feat_dict, args=hyperparams, gpu=dev.index, models=None))
    
    save_models(all_hyper_param_models, pos_classes, cfg)



        
            

