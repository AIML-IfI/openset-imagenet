import argparse
from . import approach_hierarchy
from .. import tools, openset_algos
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


class OpenMax(approach_hierarchy.FeatureApproach):
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture, used_dnn, fpr_thresholds):
        super().__init__(protocol, gpu, ku_target, uu_target,
                         model_path, log_path, oscr_path, train_cls, architecture, used_dnn, type(self).__name__, fpr_thresholds)

    def test(self, test_data, test_logits, hyperparams, model_dict):
        """
        Args:
            test_data (dict): dictionary with K representing class targets and V given by the class-specific features of the test set
            test_logits (None): dictionary with K representing class targets and V given by the class-specific logit vectors of the test set
            hyperparams (namedtuple): named tuple containing the hyperparameter options. Only the 'distance_metric' argument is actually used during inference
            model_dict (dict): dictionary that contains all the necessary information from the training procedure, including the model
        """
        logger.debug('\n')
        logger.info(f'Starting {self.approach} Testing Procedure:')

        models = model_dict['model']

        for alpha in hyperparams.alpha:

            dict_probs = self._compute_probabilities(
                test_data.keys(), test_data, hyperparams, models)
            #print(dict_probs)

            for idx, key in enumerate(dict_probs.keys()):
                assert key == list(test_logits.keys())[idx]
                assert dict_probs[key].shape[1] == test_logits[key].shape[1]
            #    if key<0: # HB just testing
            #        print("Saw a negative class which is ", key, "shapes", dict_probs[key].shape)
                probs_openmax = openset_algos.openmax_alpha(
                    dict_probs[key], test_logits[key], alpha=alpha, ignore_unknown_class=True)

                dict_probs[key] = probs_openmax
                #print("key ", key, "evt prob shape ", dict_probs[key].shape)

            all_gt, all_prob, all_predicted = self._prepare_probs_for_oscr(
                dict_probs)

            vast.tools.set_device_cpu()
            fpr, ccr, cover = vast.eval.tensor_OSRC(
                torch.tensor(all_gt), all_predicted, all_prob)

            tools.store_oscr_metrics(
                self.oscr_path / self._get_filename_oscr('test', f'{model_dict["hparam_combo"]}_alpha{alpha}', hyperparams.distance_metric), fpr, ccr, cover)
            filename = self._get_filename_oscr(
                'test', f'{model_dict["hparam_combo"]}_alpha{alpha}', hyperparams.distance_metric)
            logger.info(
                f'File with stored OSCR metrics: {filename}')

    def _validate(self, pos_classes_val, val_data, val_logits, hyperparams, hparam_combo, models):
        """
        Args:
            pos_classes_val(int, list): list of classes in the validation set being processed in current process, used for multi-processing(see source code comments in eosa/openset_algos/evm.py for details)
            val_data(dict): dictionary with K representing class targets and V given by the class-specific features of the validation set
            val_logits (dict): dictionary with K representing class targets and V given by the class-specific logit vectors of the validation set
            hyperparams(namedtuple): named tuple containing the hyperparameter options. Only the 'distance_metric' argument is actually used during inference
            hparam_combo (string): hyperparameter combination that represents the model which is evaluated
            models(dict): dictionary containing the collated OpenMax model that is associated with a specific hyperparameter combination. K representing training class targets and V given by collated OpenMax model as computed in training
        """

        for alpha in hyperparams.alpha:

            dict_probs = self._compute_probabilities(
                pos_classes_val, val_data, hyperparams, models)

            dict_probs_rev = {}
            for idx, key in enumerate(dict_probs.keys()):
                assert key == list(val_logits.keys())[idx]
                assert dict_probs[key].shape[1] == val_logits[key].shape[1]

                probs_openmax = openset_algos.openmax_alpha(
                    evt_probs=dict_probs[key].clone().detach(), activations=val_logits[key].clone().detach(), alpha=alpha, ignore_unknown_class=True)
                dict_probs_rev[key] = probs_openmax

            all_gt, all_prob, all_predicted = self._prepare_probs_for_oscr(
                dict_probs_rev)

            vast.tools.set_device_cpu()
            fpr, ccr, cover = vast.eval.tensor_OSRC(
                torch.tensor(all_gt), all_predicted, all_prob)

            tools.store_oscr_metrics(
                self.oscr_path / self._get_filename_oscr('eval', f'{hparam_combo}_alpha{alpha}', hyperparams.distance_metric), fpr, ccr, cover)

            self._get_ccr_at_fpr(hparam_combo, self.oscr_path /
                                 ('CCR@FPR_' + self._get_filename_oscr('eval', f'{hparam_combo}_alpha{alpha}', hyperparams.distance_metric)), fpr, ccr)
            filename = self._get_filename_oscr(
                'eval', f'{hparam_combo}_alpha{alpha}', hyperparams.distance_metric)
            logger.info(
                f'File with stored OSCR metrics: {filename}')


def openmax_hyperparams(tailsize, dist_mult, translate_amount, dist_metric, alpha):

    hparam_string = f'--tailsize {" ".join(str(e) for e in tailsize)} --distance_multiplier {" ".join(str(e) for e in dist_mult)} --distance_metric {dist_metric}'

    parser = argparse.ArgumentParser()
    parser, _ = getattr(opensetAlgos, 'OpenMax_Params')(parser)

    ns_obj = parser.parse_args(hparam_string.split())
    ns_obj.alpha = alpha
    return ns_obj
