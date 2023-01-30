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
from .util import calculate_oscr





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
    probability_vector = F.softmax(probability_vector, dim=1)

    if ignore_unknown_class:
        probs_kkc = probability_vector[:, 1:].clone().detach()
        assert probs_kkc.shape == activations.shape
        #print("Openmax Alpha is returning probs_kkc and shape ", probs_kkc.shape)
        return probs_kkc

    # Line 8
    prediction_score, predicted_class = torch.max(probability_vector, dim=1)
    # Line 9
    prediction_score[predicted_class == 0] = -1.0
    predicted_class = predicted_class - 1

    return predicted_class, prediction_score


def compute_adjust_probs(gt, logits, features, scores, model_dict, cfg, hyperparams, alpha_index=0):
    #alpha index is used to indicate which alpha value to be tested in case there are multiple of them given
    #arrange for openmax inference/alpha
    gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)
    kkc = collect_pos_classes(gt)
    #targets, features, logits = postprocess_train_data(gt, features, logits)
    #print(kkc)
    pos_classes = collect_pos_classes(gt)
    feat_dict, logit_dict = compose_dicts(gt, features, logits)
    feat_dict = {k: v.double() for k, v in feat_dict.items()}

    print(hyperparams)

    #for alpha in hyperparams.alpha:
    probabilities = list(getattr(opensetAlgos, f'{vars(cfg.alg_dict)[cfg.algorithm.type]}_Inference')(pos_classes_to_process=feat_dict.keys(), features_all_classes=feat_dict, args=hyperparams, gpu=cfg.gpu, models=model_dict['model']))
    dict_probs = dict(list(zip(*probabilities))[1])
    for idx, key in enumerate(dict_probs.keys()):
        print(list(logit_dict.keys())[idx], key, type(list(logit_dict.keys())[idx]), type(key))
        assert key == list(logit_dict.keys())[idx]
        assert dict_probs[key].shape[1] == logit_dict[key].shape[1]
        probs_openmax = openset_algos.openmax_alpha(dict_probs[key], logit_dict[key], alpha=hyperparams.alpha[alpha_index], ignore_unknown_class=True)
        dict_probs[key] = probs_openmax

    all_probs = []
    for key in range(len(kkc)):
        #print(key, dict_probs[key].shape[-1])
        all_probs.extend(dict_probs[key].tolist())

    for key in [-1, -2]:
        if key in dict_probs.keys():
            print(key, 'shape:', dict_probs[key].shape[-1])
            all_probs.extend(dict_probs[key].tolist())

    return all_probs

def compute_probs(gt, logits, features, scores, model_dict, cfg, hyperparams):
    #arrange for EVM
    gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)
    kkc = collect_pos_classes(gt)
    #targets, features, logits = postprocess_train_data(gt, features, logits)
    print('compute_probs:', kkc)
    pos_classes = collect_pos_classes(gt)
    feat_dict, logit_dict = compose_dicts(gt, features, logits)
    feat_dict = {k: v.double() for k, v in feat_dict.items()}

    print(hyperparams)


    probabilities = list(getattr(opensetAlgos, f'{vars(cfg.alg_dict)[cfg.algorithm.type]}_Inference')(pos_classes_to_process=feat_dict.keys(),features_all_classes=feat_dict, args=hyperparams, gpu=cfg.gpu, models=model_dict['model']))
    dict_probs = dict(list(zip(*probabilities))[1])


    all_probs = []
    for key in range(len(kkc)):
        #print(key, dict_probs[key].shape[-1])
        all_probs.extend(dict_probs[key].tolist())

    for key in [-1, -2]:
        if key in dict_probs.keys():
            print(key, dict_probs[key].shape[-1])
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
            model_name = f'{cfg.loss.type}_{cfg.algorithm.type}_{key}_{cfg.algorithm.distance_metric}.pkl'

            file_handler = open(pathlib.Path(cfg.output_directory) / model_name, 'wb')

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

'''
def validate(all_hyper_param_models,pos_classes, cfg):
    # integrating returned models in a data structure as required by <self.approach>_Inference()
    hparam_combo_to_model = defaultdict(list)
    for i in range(len(all_hyper_param_models)):
        hparam_combo_to_model[all_hyper_param_models[i][0]].append(all_hyper_param_models[i][1])

    logger.info(f'Trained models associated with hyperparameters: {list(hparam_combo_to_model.keys())}')
    for key in hparam_combo_to_model:
        hparam_combo_to_model[key] = dict(hparam_combo_to_model[key])
'''

def get_ccr_at_fpr(hparam_combo, path, fpr, ccr, cfg):
    """
    Args:
        hparam_combo (string): hyperparameter combination that represents the model whose oscr metrics are analysed
        path (string, pathlib.Path): path to the csv file where oscr metrics are stored
        fpr (torch.Tensor): tensor representing the (oscr) false positive rate
        ccr (torch.Tensor): tensor representing the (oscr) correct classification rate
        fpr_thresholds (list<float>): list of false positive rate (fpr) thresholds that are used to select the best model in the validation procedure in a "ccr@fpr" fashion (see Dhamija et al., 2018)
    """
    assert fpr.shape == ccr.shape, 'shape mismatch between torch tensors'

    values = [('model', 'fpr_threshold', 'idx_closest_fpr', 'fpr', 'ccr')]

    for e in cfg.fpr_thresholds:
        threshold = torch.Tensor([e]*fpr.shape[0])
        #print(torch.abs((fpr - threshold)))
        _, min_idx = torch.min(torch.abs((fpr - threshold)), -1)
        values.append((hparam_combo, e, min_idx.item(),
                        fpr[min_idx].item(), ccr[min_idx].item()))

    with open(path, 'w') as file_out:
        writer = csv.writer(file_out)
        for e in values:
            writer.writerow(e)

    print("CCR@FPR written in", path)
    return values

def get_avail_ccr_at_fpr(hparam_combo, path, fpr, ccr, cfg):
    """
    Args:
        hparam_combo (string): hyperparameter combination that represents the model whose oscr metrics are analysed
        path (string, pathlib.Path): path to the csv file where oscr metrics are stored
        fpr (torch.Tensor): tensor representing the (oscr) false positive rate
        ccr (torch.Tensor): tensor representing the (oscr) correct classification rate
        fpr_thresholds (list<float>): list of false positive rate (fpr) thresholds that are used to select the best model in the validation procedure in a "ccr@fpr" fashion (see Dhamija et al., 2018)
    """
    assert fpr.shape == ccr.shape, 'shape mismatch between torch tensors'

    values = [('model', 'fpr_threshold', 'idx_closest_fpr', 'fpr', 'ccr')]

    for index, e in enumerate(cfg.fpr_thresholds):
        #threshold = torch.Tensor([e]*fpr.shape[0])
        lower_value = []
        lower_index = []
        for i, f in enumerate(fpr):
            if f<e:
                lower_value.append(f)
                lower_index.append(i)
        if len(lower_value)>0:
            max_index= np.argmax(np.array(lower_value))
            f_index = lower_index[max_index]

        #print(torch.abs((fpr - threshold)))
        #_, min_idx = torch.min(torch.abs((fpr - threshold)), -1)
            values.append((hparam_combo, e, f_index, fpr[f_index].item(), ccr[f_index].item()))

    with open(path, 'w') as file_out:
        writer = csv.writer(file_out)
        for e in values:
            writer.writerow(e)

    print("CCR@FPR written in", path)
    return values

def validate(gt, logits, features, scores, model_dict, hyperparams, cfg):

    #model_dict is trained model with openmax or evm which has weibull distributions saved
    #hyperparams are hyperparameters being stored for openmax and evm such as distance multiplier, distance metric, alpha etc.



    #gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)

    #feat_dict, _ = compose_dicts(gt, features, logits)


    if cfg.algorithm.type == 'openmax':
        #scores are being adjusted her through openmax alpha
        print("adjusting probabilities for openmax with alpha on validation set)")
        for index, _ in enumerate(hyperparams.alpha):
            scores = compute_adjust_probs(gt, logits, features, scores, model_dict, cfg, hyperparams, index)
            ccr, fpr = calculate_oscr(gt, np.array(scores), unk_label=-1)
            get_avail_ccr_at_fpr(model_dict['hparam_combo'], cfg.output_directory/('CCR@FPR_' + f"{cfg.loss.type}_{cfg.algorithm.type}_"+ model_dict['hparam_combo'] + "_alpha_" + str(hyperparams.alpha[index]) + "_"+hyperparams.distance_metric + ".csv"), torch.Tensor(fpr), torch.Tensor(ccr), cfg)
    elif cfg.algorithm.type=='evm':
        print("computing probabilities for evm on validation set")
        scores = compute_probs(gt, logits, features, scores, model_dict, cfg, hyperparams)
        ccr, fpr = calculate_oscr(gt, np.array(scores), unk_label=-1)
        print(ccr, fpr)
        get_avail_ccr_at_fpr(model_dict['hparam_combo'], cfg.output_directory/('CCR@FPR_' + f"{cfg.loss.type}_{cfg.algorithm.type}_"+ model_dict['hparam_combo'] + "_"+hyperparams.distance_metric + ".csv"), torch.Tensor(fpr), torch.Tensor(ccr), cfg)



    #file_path = args.output_directory / f"{args.loss}_{cfg.algorithm.type}_val_arr{suffix}.npz"
    #np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    #print(f"Target labels, logits, features and scores saved in: {file_path}")

    #get_ccr_at_fpr(hparam_combo, args.output_directory, fpr, ccr, cfg)


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
        #HB things might be different for garbage
        if cfg.loss.type == "garbage":
            # Only change the unknown label of the training dataset
            train_ds.replace_negative_label()
        else:
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

    n_classes = train_ds.label_count
    # Create the model
    model = ResNet50(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)
    device(model)
    logger.debug('\n')

    print("GPU:", cfg.gpu, cfg.algorithm.type, cfg.loss.type)

    print('will write in:', cfg.output_directory)

    # load checkpoint
    base_model_path = cfg.model_path.format(cfg.output_directory, cfg.loss.type, "threshold", "curr")
    start_epoch, best_score = load_checkpoint(model, checkpoint=base_model_path)

    print(f"Loaded {base_model_path} and taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)

    if cfg.algorithm.type== 'openmax':
        hyperparams = openmax_hyperparams(cfg.algorithm.tailsize, cfg.algorithm.distance_multiplier, cfg.algorithm.translateAmount, cfg.algorithm.distance_metric, cfg.algorithm.alpha)
    elif cfg.algorithm.type == 'evm':
        hyperparams = evm_hyperparams(cfg.algorithm.tailsize, cfg.algorithm.cover_threshold,cfg.algorithm.distance_multiplier, cfg.algorithm.distance_metric, cfg.algorithm.chunk_size)

    print(f'{(cfg.algorithm.type)}_hyperparams finished')


    #hyperparams = getattr(approaches, f'{cfg.alg}_hyperparams')(*alg_hyperparameters)


    print("========== Training  ==========")

    print("Feature extraction on training data:")

    if not cfg.parallel:
        train_loader = tqdm.tqdm(train_loader)
    # extracting arrays for training data
    gt, logits, features, scores = get_arrays(
            model=model,
            loader=train_loader
            )

    if cfg.loss.type=='garbage':
        biggest_label = np.sort(np.unique(gt))[-1]
        print(biggest_label)
        gt[gt==biggest_label] = -1
        logits = logits[:,:-1]
        features = features[:,:-1]

    gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)

    kkc = collect_pos_classes(gt) #HBfor garbage this returns class 116 which was for negative classees.

    targets, features, logits = postprocess_train_data(gt, features, logits)
    #print(kkc)
    pos_classes = collect_pos_classes(targets)

    feat_dict, _ = compose_dicts(targets, features, logits)

    #approach = getattr(approaches, cfg.alg)

    #print(approach)

    logger.debug('\n')
    logger.info(f'Starting {cfg.algorithm.type} Training Procedure:')

    training_fct = {
        'openmax' : opensetAlgos.OpenMax_Training,
        'evm' : opensetAlgos.EVM_Training

    }[cfg.algorithm.type]

    #performs training on all parameter combinations
    #Training method returns iterator over (hparam_combo, (class, {model}))
    all_hyper_param_models = list(training_fct(
        pos_classes_to_process=pos_classes, features_all_classes=feat_dict, args=hyperparams, gpu=cfg.gpu, models=None))

    save_models(all_hyper_param_models, pos_classes, cfg)
    logger.info(f'{cfg.algorithm.type} Training Finished')
