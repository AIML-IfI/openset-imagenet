from . import approach_hierarchy
from .. import tools
from collections import namedtuple
import logging
import numpy as np
import random
import torch
import vast
from vast import opensetAlgos

random.seed(0)

# instantiate module logger
logger = logging.getLogger(__name__)


class EVM(approach_hierarchy.FeatureApproach):
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture, used_dnn, fpr_thresholds):
        super().__init__(protocol, gpu, ku_target, uu_target,
                         model_path, log_path, oscr_path, train_cls, architecture, used_dnn, type(self).__name__, fpr_thresholds)

    def test(self, test_data, test_logits, hyperparams, model_dict):
        """
        Args:
            test_data (dict): dictionary with K representing class targets and V given by the class-specific features of the test set
            test_logits (None): None since EVM does not need logit vectors
            hyperparams (namedtuple): named tuple containing the hyperparameter options. Only the 'distance_metric' argument is actually used during inference
            model_dict (dict): dictionary that contains all the necessary information from the training procedure, including the model
        """
        _ = test_logits

        logger.debug('\n')
        logger.info(f'Starting {self.approach} Testing Procedure:')

        model = model_dict['model']

        dict_probs = self._compute_probabilities(
            list(test_data.keys()), test_data, hyperparams, model)

        all_gt, all_prob, all_predicted = self._prepare_probs_for_oscr(
            dict_probs)

        vast.tools.set_device_cpu()
        fpr, ccr, cover = vast.eval.tensor_OSRC(
            torch.tensor(all_gt), all_predicted, all_prob)

        tools.store_oscr_metrics(
            self.oscr_path / self._get_filename_oscr('test', model_dict["hparam_combo"], hyperparams.distance_metric), fpr, ccr, cover)
        logger.info(
            f'File with stored OSCR metrics: {self._get_filename_oscr("test", model_dict["hparam_combo"], hyperparams.distance_metric)}')

    def _validate(self, pos_classes_val, val_data, val_logits, hyperparams, hparam_combo, models):
        """
        Args:
            pos_classes_val (int, list): list of classes in the validation set being processed in current process, used for multi-processing(see source code comments in eosa/openset_algos/evm.py for details)
            val_data (dict): dictionary with K representing class targets and V given by the class-specific features of the validation set
            val_logits (None): None since EVM does not need logit vectors
            hyperparams (namedtuple): named tuple containing the hyperparameter options. Only the 'distance_metric' argument is actually used during inference
            hparam_combo (string): hyperparameter combination that represents the model which is evaluated
            models (dict): dictionary containing the EVM model that is associated with a specific hyperparameter combination. K representing training class targets and V given by class-specific EVM model as computed in training
        """
        _ = val_logits

        dict_probs = self._compute_probabilities(
            pos_classes_val, val_data, hyperparams, models)

        all_gt, all_prob, all_predicted = self._prepare_probs_for_oscr(
            dict_probs)

        vast.tools.set_device_cpu()
        fpr, ccr, cover = vast.eval.tensor_OSRC(
            torch.tensor(all_gt), all_predicted, all_prob)

        tools.store_oscr_metrics(
            self.oscr_path / self._get_filename_oscr('eval', hparam_combo, hyperparams.distance_metric), fpr, ccr, cover)

        self._get_ccr_at_fpr(hparam_combo, self.oscr_path /
                             ('CCR@FPR_' + self._get_filename_oscr('eval', hparam_combo, hyperparams.distance_metric)), fpr, ccr)
        logger.info(
            f'File with stored OSCR metrics: {self._get_filename_oscr("eval", hparam_combo, hyperparams.distance_metric)}')


def evm_hyperparams(tailsize, cover_thres, dist_mult, dist_metric, chunk):
    hyperparams = namedtuple('EVM_Hyperparams', [
        'tailsize', 'cover_threshold', 'distance_multiplier', 'distance_metric', 'chunk_size'])
    return hyperparams(tailsize, cover_thres, dist_mult, dist_metric, chunk)
