import sys
import pathlib
from collections import defaultdict
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from vast.tools import set_device_gpu, set_device_cpu, device, _device
import vast
from loguru import logger
from .dataset import ImagenetDataset
from .model import ResNet50
import tqdm
import pandas as pd
import pickle
from vast import opensetAlgos
from .train import set_seeds, load_checkpoint, get_arrays
import argparse
from collections import namedtuple
import csv
from .util import calculate_oscr, NameSpace


def get_training_function(type):
    return {
        'openmax' : opensetAlgos.OpenMax_Training,
        'evm' : opensetAlgos.EVM_Training
    }[type]

def get_inference_function(type):
    return {
        'openmax' : opensetAlgos.OpenMax_Inference,
        'evm' : opensetAlgos.EVM_Inference
    }[type]


def get_param_string(type, **kwargs):
    if type == "openmax":
        # CF: https://github.com/Vastlab/vast/blob/main/vast/opensetAlgos/openmax.py#L100
        return f"TS_{kwargs['tailsize']}_DM_{kwargs['distance_multiplier']:.2f}"
    elif type == "evm":
        # CF: https://github.com/Vastlab/vast/blob/main/vast/opensetAlgos/EVM.py#L294
        return f"TS_{kwargs['tailsize']}_DM_{kwargs['distance_multiplier']:.2f}_CT_{kwargs['cover_threshold']:.2f}"


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
    logger.debug(f'Number of samples included in the dict: {count_feat}')
    logger.debug(f'Number of classes (i.e. # dict keys): {len(list(feat_dict.keys()))}')
    return feat_dict, logit_dict

def postprocess_train_data(targets, features, logits):
    # Note: OpenMax uses only the training samples that get correctly classified by the
          # underlying, extracting DNN to train its model.logger.debug('\n')

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


def openmax_alpha(
    evt_probs, activations, alpha=1, run_paper_version=True, ignore_unknown_class=False, *args, **kwargs
):
    """
    Algorithm 2 OpenMax probability estimation with rejection of
    unknown or uncertain inputs.
    Require: Activation vector for v(x) = v1(x), . . . , vN (x)
    Require: means µj and libMR models ρj = (τi, λi, κi)
    Require: α, the numer of “top” classes to revise
    1: Let s(i) = argsort(vj (x)); Let ωj = 1
    2: for i = 1, . . . , α do
    3:     ωs(i)(x) = 1 − ((α−i)/α)*e^−((||x−τs(i)||/λs(i))^κs(i))
    4: end for
    5: Revise activation vector vˆ(x) = v(x) ◦ ω(x)
    6: Define vˆ0(x) = sum_i vi(x)(1 − ωi(x)).
    7:     Pˆ(y = j|x) = eˆvj(x)/sum_{i=0}_N eˆvi(x)
    8: Let y∗ = argmaxj P(y = j|x)
    9: Reject input if y∗ == 0 or P(y = y∗|x) < ǫ
    """
    # convert weibull CDF probabilities from knownness per class to unknownness per class
    per_class_unknownness_prob = 1 - evt_probs

    # Line 1
    sorted_activations, indices = torch.sort(
        activations, descending=True, dim=1)
    weights = torch.ones(activations.shape[0], activations.shape[1])

    # Line 2-4
    weights[:, :alpha] = torch.arange(1, alpha + 1, step=1)
    if run_paper_version:
        weights[:, :alpha] = (alpha - weights[:, :alpha]) / alpha
    else:
        # The version in the code is slightly different from the algorithm mentioned in the paper
        weights[:, :alpha] = ((alpha + 1) - weights[:, :alpha]) / alpha
    weights[:, :alpha] = 1 - weights[:, :alpha] * torch.gather(
        per_class_unknownness_prob, 1, indices[:, :alpha]
    )

    # Line 5
    revisted_activations = sorted_activations * weights
    # Line 6
    unknowness_class_prob = torch.sum(
        sorted_activations * (1 - weights), dim=1)
    revisted_activations = torch.scatter(
        torch.ones(revisted_activations.shape), 1, indices, revisted_activations
    )
    probability_vector = torch.cat(
        [unknowness_class_prob[:, None], revisted_activations], dim=1
    )

    # Line 7
    probability_vector = torch.nn.functional.softmax(probability_vector, dim=1)

    if ignore_unknown_class:
        probs_kkc = probability_vector[:, 1:].clone().detach()
        assert probs_kkc.shape == activations.shape
        return probs_kkc

    # Line 8
    prediction_score, predicted_class = torch.max(probability_vector, dim=1)
    # Line 9
    prediction_score[predicted_class == 0] = -1.0
    predicted_class = predicted_class - 1

    return predicted_class, prediction_score


def compute_adjust_probs(gt, logits, features, scores, model_dict, algorithm, gpu_index, hyperparams, alpha):
    #alpha index is used to indicate which alpha value to be tested in case there are multiple of them given
    # arrange for openmax inference/alpha
    gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)
    pos_classes = collect_pos_classes(gt)
    feat_dict, logit_dict = compose_dicts(gt, features, logits)
    feat_dict = {k: v.double() for k, v in feat_dict.items()}

    #for alpha in hyperparams.alpha:
    probabilities = list(get_inference_function(algorithm)(pos_classes_to_process=feat_dict.keys(), features_all_classes=feat_dict, args=hyperparams, gpu=gpu_index, models=model_dict['model']))
    dict_probs = dict(list(zip(*probabilities))[1])
    for idx, key in enumerate(dict_probs.keys()):
        assert key == list(logit_dict.keys())[idx]
        assert dict_probs[key].shape[1] == logit_dict[key].shape[1]
        probs_openmax = openmax_alpha(dict_probs[key], logit_dict[key], alpha=alpha, ignore_unknown_class=True)
        dict_probs[key] = probs_openmax

    all_probs = []
    for key in range(len(pos_classes)):
        all_probs.extend(dict_probs[key].tolist())

    for key in [-1, -2]:
        if key in dict_probs.keys():
            all_probs.extend(dict_probs[key].tolist())

    return all_probs

def compute_probs(gt, logits, features, scores, model_dict, algorithm, gpu_index, hyperparams):
    #arrange for EVM
    gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)
    pos_classes = collect_pos_classes(gt)
    feat_dict, logit_dict = compose_dicts(gt, features, logits)
    feat_dict = {k: v.double() for k, v in feat_dict.items()}

    probabilities = list(get_inference_function(algorithm)(pos_classes_to_process=feat_dict.keys(),features_all_classes=feat_dict, args=hyperparams, gpu=gpu_index, models=model_dict['model']))
    dict_probs = dict(list(zip(*probabilities))[1])

    all_probs = []
    for key in range(len(pos_classes)):
        all_probs.extend(dict_probs[key].tolist())

    for key in [-1, -2]:
        if key in dict_probs.keys():
            all_probs.extend(dict_probs[key].tolist())

    return all_probs

def save_models(all_hyper_param_models,pos_classes, cfg):
        # integrating returned models in a data structure as required by <self.approach>_Inference()
        hparam_combo_to_model = defaultdict(list)

        for i in range(len(all_hyper_param_models)):
            hparam_combo_to_model[all_hyper_param_models[i][0]].append(all_hyper_param_models[i][1])

        logger.info(f'Trained models associated with hyperparameters: {list(hparam_combo_to_model.keys())}')
        for key in hparam_combo_to_model:
            hparam_combo_to_model[key] = dict(hparam_combo_to_model[key])

            # store models per hyperparameter combination as a (hparam_combo, model)-tuple
            model_name = cfg.algorithm.output_model_path.format(
                cfg.output_directory, cfg.loss.type, cfg.algorithm.type, key, cfg.algorithm.distance_metric
            )

            file_handler = open(model_name, 'wb')

            obj_serializable = {'approach_train': cfg.algorithm.type, 'model_name': model_name,
                    'hparam_combo': key, 'distance_metric': cfg.algorithm.distance_metric, 'model':  hparam_combo_to_model[key]}


            pickle.dump(obj_serializable, file_handler)

            """
            Important: Since the <approach>_Inference() function in the vast package sorts the
            keys of the collated model, the semantic of the returned probabilities depends on
            the type of the dictionary keys. For example, when sorting is applied on the 'stringified'
            integer classes, the column indices of the returned probabilities tensor do not necessarily
            correspond with the integer class targets. Hence, the assertion for integer type below.
            """
            assert sum([isinstance(k, int) for k in hparam_combo_to_model[key].keys()]) == len(
                list(hparam_combo_to_model[key].keys())), 'dictionary\'s keys are not of type "int"'

        """
        SANITY CHECKS
        """
        assert len(set([el[0] for el in all_hyper_param_models])) == len(
            hparam_combo_to_model.keys()), 'missing entries for hyperparameter combinations'
        assert [(el == len(pos_classes)) for el in [len(hparam_combo_to_model[k].keys())
                                                    for k in hparam_combo_to_model.keys()]], 'model misses training class(es)'



def validate(gt, logits, features, scores, model_dict, hyperparams, cfg):

    if cfg.algorithm.type == 'openmax':
        #scores are being adjusted her through openmax alpha
        print("adjusting probabilities for openmax with alpha on validation set)")
        for alpha in hyperparams.alpha:
            scores = compute_adjust_probs(gt, logits, features, scores, model_dict, cfg.algorithm.type, cfg.gpu, hyperparams, alpha)
            ccr, fpr = calculate_oscr(gt, np.array(scores), unk_label=-1)
            get_avail_ccr_at_fpr(model_dict['hparam_combo'], cfg.output_directory/('CCR@FPR_' + f"{cfg.loss.type}_{cfg.algorithm.type}_"+ model_dict['hparam_combo'] + "_alpha_" + str(alpha) + "_"+hyperparams.distance_metric + ".csv"), torch.Tensor(fpr), torch.Tensor(ccr), cfg)
    elif cfg.algorithm.type=='evm':
        print("computing probabilities for evm on validation set")
        scores = compute_probs(gt, logits, features, scores, model_dict, cfg.algorithm.type, cfg.gpu, hyperparams)
        ccr, fpr = calculate_oscr(gt, np.array(scores), unk_label=-1)
        print(ccr, fpr)
        get_avail_ccr_at_fpr(model_dict['hparam_combo'], cfg.output_directory/('CCR@FPR_' + f"{cfg.loss.type}_{cfg.algorithm.type}_"+ model_dict['hparam_combo'] + "_"+hyperparams.distance_metric + ".csv"), torch.Tensor(fpr), torch.Tensor(ccr), cfg)



def openmax_hyperparams(tailsize, dist_mult, translate_amount, dist_metric, alpha):

    return NameSpace(dict(
        tailsize=tailsize, distance_multiplier=dist_mult, distance_metric=dist_metric, alpha=alpha
    ))

def evm_hyperparams(tailsize, cover_thres, dist_mult, dist_metric, chunk):
    return NameSpace(dict(
        tailsize=tailsize, distance_multiplier=dist_mult, cover_threshold=cover_thres, distance_metric=dist_metric, chunk_size=chunk
    ))


def worker(cfg):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        cfg (NameSpace): Configuration of the experiment
    """
    # referencing best score and setting seeds
    set_seeds(cfg.seed)

    # Configure logger. Log only on first process. Validate only on first process.
#    msg_format = "{time:DD_MM_HH:mm} {message}"
    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink= pathlib.Path(cfg.output_directory) / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')

    # Set image transformations
    train_tr = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    # create datasets
    train_file = pathlib.Path(cfg.data.train_file.format(cfg.protocol))


    if train_file.exists() :
        train_ds = ImagenetDataset(
            csv_file=train_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=train_tr
        )
        # We train only on positive labels for now (we might incorporate negative labels as extra negatives for EVM later)
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
        logger.warning("No GPU device selected, feature extraction will be slow")
        set_device_cpu()

    n_classes = train_ds.label_count + (1 if cfg.loss.type=="garbage" else 0)
    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)
    device(model)

    # load checkpoint
    base_model_path = cfg.model_path.format(cfg.output_directory, cfg.loss.type, "threshold", "curr")
    start_epoch, best_score = load_checkpoint(model, checkpoint=base_model_path)

    logger.info(f"Loaded {base_model_path} and taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)

    if cfg.algorithm.type== 'openmax':
        hyperparams = openmax_hyperparams(cfg.algorithm.tailsize, cfg.algorithm.distance_multiplier, cfg.algorithm.translateAmount, cfg.algorithm.distance_metric, cfg.algorithm.alpha)
    elif cfg.algorithm.type == 'evm':
        hyperparams = evm_hyperparams(cfg.algorithm.tailsize, cfg.algorithm.cover_threshold,cfg.algorithm.distance_multiplier, cfg.algorithm.distance_metric, cfg.algorithm.chunk_size)

    logger.info("Feature extraction on training data:")

    # extracting arrays for training data
    gt, logits, features, scores = get_arrays(
            model=model,
            loader=train_loader,
            garbage=cfg.loss.type=="garbage",
            pretty=not cfg.parallel
    )


    gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)

    targets, features, logits = postprocess_train_data(gt, features, logits)
    pos_classes = collect_pos_classes(targets)

    feat_dict, _ = compose_dicts(targets, features, logits)

    logger.debug('\n')
    logger.info(f'Starting {cfg.algorithm.type} Training Procedure:')

    training_fct = get_training_function(cfg.algorithm.type)

    #performs training on all parameter combinations
    #Training method returns iterator over (hparam_combo, (class, {model}))
    all_hyper_param_models = list(training_fct(
        pos_classes_to_process=pos_classes, features_all_classes=feat_dict, args=hyperparams, gpu=cfg.gpu, models=None))

    save_models(all_hyper_param_models, pos_classes, cfg)
    logger.info(f'{cfg.algorithm.type} Training Finished')
