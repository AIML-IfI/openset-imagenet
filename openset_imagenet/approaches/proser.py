
from collections import defaultdict
from . import approach_hierarchy
from argparse import Namespace
from .. import architectures, tools
from .. import openset_algos as os_alg
import logging
import numpy as np
import os
from pathlib import Path
import random
import time
import torch
from tqdm import tqdm
import vast

random.seed(0)

# instantiate module logger
logger = logging.getLogger(__name__)


class Proser(approach_hierarchy.RootApproach):
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture, epochs_finetune, basis_model, lambda1, lambda2, alpha, no_dummy_clfs, compute_bias):
        super().__init__(protocol, gpu, ku_target, uu_target,
                         model_path, log_path, oscr_path, train_cls, architecture)
        """
        Args:
            epochs_finetune (int): number of training epochs
            basis_model (dict): dictionary that contains all the necessary information associated with the basis model
            lambda1 (float): parameter used for loss configuration (-> see https://github.com/zhoudw-zdw/CVPR21-Proser/blob/f5b53b90509e6460a783a0baa72bda52364810fd/proser_unknown_detection.py#L253)
            lambda2 (float): parameter used for loss configuration (-> see https://github.com/zhoudw-zdw/CVPR21-Proser/blob/f5b53b90509e6460a783a0baa72bda52364810fd/proser_unknown_detection.py#L254)
            alpha (float): parameter used for initializing the Beta distribution (-> see https://github.com/zhoudw-zdw/CVPR21-Proser/blob/f5b53b90509e6460a783a0baa72bda52364810fd/proser_unknown_detection.py#L39)
            no_dummy_clfs (int): number of dummy classifiers to be added to the basis model
        """
        self.epochs_finetune = epochs_finetune
        self.epoch_current = None
        self.basis_model_dict = basis_model

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha

        self.no_dummy_clfs = no_dummy_clfs
        self.compute_bias = compute_bias

        self.args_proser = self._get_args_proser()
        self.eval_metric_best = 0.
        self.epoch_best = 0
        self.model_path_best = None
        self.training_metric_epochs = defaultdict(list)
        self.validation_metric_epochs = defaultdict(list)

    def train(self, train_loader, val_loader):
        # load the basis model and extend it by the dummy classifiers
        model = getattr(architectures, self.basis_model_dict['instance']['architecture'])(
            self.basis_model_dict['instance']['df_dim'], self.basis_model_dict['instance']['num_cls'])
        model.load_state_dict(self.basis_model_dict['state_dict'])
        model.clf2 = torch.nn.Linear(
            self.basis_model_dict['instance']['df_dim'], self.no_dummy_clfs)
        model.to(self.device)

        t0_training = time.time()

        tools.get_cuda_info(self.device)
        tools.get_model_status(model)
        self._set_seeds(self.device)
        logger.info(f'Proser arguments: {self.args_proser}')

        for epoch in range(self.epochs_finetune):
            t0_epoch = time.time()
            self.epoch_current = epoch+1
            # calling Proser training method
            bias_epoch, loss_epoch, accuracy_epoch = os_alg.finetune_proser(
                model, epoch, self.args_proser, train_loader, self.device)

            self.training_metric_epochs['loss'].append(loss_epoch)
            self.training_metric_epochs['accuracy'].append(accuracy_epoch)
            logger.info(
                f'Avg. Epoch Loss: {loss_epoch} - Epoch Accuracy: {accuracy_epoch}')

            logger.debug('\n')
            logger.info(
                f'Starting {type(self).__name__} Evaluation: Epoch {epoch+1}')
            self._validate(model, val_loader, bias_epoch)

            logger.info(
                f'Training time for epoch {epoch+1}:\t {((time.time() - t0_epoch) // 60):.0f}m {((time.time() - t0_epoch) % 60):.0f}s')

        logger.info(
            f'Training time:\t {((time.time() - t0_training) // 60):.0f}m')
        logger.info(
            f'Training Metrics per Epoch: {self.training_metric_epochs}')
        logger.info(
            f'Validation Metrics per Epoch: {self.validation_metric_epochs}')
        logger.info(
            f'Best Validation Metric: {self.eval_metric_best} (epoch {self.epoch_best})')
        logger.info(f'Path to Best Model: {self.model_path_best}')

    def test(self, test_loader, model_dict):
        """
        Args:
            test_loader (torch.utils.data.DataLoader): object containing the testing data
            model_dict (dict): dictionary that contains all the necessary information from the training procedure, including the state_dict
        """
        logger.debug('\n')
        logger.info(f'Starting {type(self).__name__} Testing Procedure:')

        logger.info(
            f'Best training performance of loaded model: {model_dict["eval_metric"]} = {model_dict["eval_metric_opt"]:.6f}, achieved in training epoch {model_dict["epoch_opt"]}')

        self._set_seeds(self.device)

        model = getattr(architectures, self.basis_model_dict['instance']['architecture'])(
            self.basis_model_dict['instance']['df_dim'], self.basis_model_dict['instance']['num_cls'])
        model.clf2 = torch.nn.Linear(
            self.basis_model_dict['instance']['df_dim'], model_dict['instance']['no_dummy_clfs'])
        model.load_state_dict(model_dict['state_dict'])
        model.to(self.device)

        logits_all, targets_all = self._extract(
            model, test_loader, float(model_dict['bias']))

        all_gt, all_prob, all_predicted = self._prepare_probs_for_oscr(
            logits_all, targets_all)

        vast.tools.set_device_cpu()
        fpr, ccr, cover = vast.eval.tensor_OSRC(
            all_gt, all_predicted, all_prob)

        filename = f'p{self.protocol}_{type(self).__name__.lower()}_test_{Path(self._get_model_name()).stem}_oscr_values.csv'
        tools.store_oscr_metrics(
            self.oscr_path / filename, fpr, ccr, cover)
        logger.info(
            f'File with stored OSCR metrics: {filename}')

    def _extract(self, net, dataloader, bias):
        logits_all = []
        targets_all = []

        with torch.no_grad():
            net.eval()
            for x, t in tqdm(dataloader):
                x, t = x.to(self.device), t.to(self.device)

                dummylogit = os_alg.dummypredict(net, x, self.args_proser)
                logits_known, _ = net(x)
                logits = torch.cat((logits_known, dummylogit+bias), dim=1)

                logits_all.extend(logits.tolist())
                targets_all.extend(t.tolist())

        return torch.tensor(logits_all), torch.tensor(targets_all)

    def _get_args_proser(self):
        args = Namespace()
        # based on lr of basis model (-> see https://gitlab.ifi.uzh.ch/aiml/projects/placeholder-open-set/-/blob/main/proser_unknown_detection.py#L111)
        args.lr = self.basis_model_dict['instance']['optimizer']['lr']
        args.backbone = self.architecture
        args.known_class = self.basis_model_dict['instance']['num_cls']
        args.lamda1 = self.lambda1
        args.lamda2 = self.lambda2
        args.alpha = self.alpha
        args.dummynumber = self.no_dummy_clfs
        args.compute_bias = self.compute_bias
        return args

    def _get_model_name(self):
        return f'p{self.protocol}_traincls({"+".join(self.train_classes)})_{type(self).__name__.lower()}_dummy{self.no_dummy_clfs}_epochsfine{self.epochs_finetune}_λ({self.lambda1}+{self.lambda2})_α{self.alpha}_bias({self.compute_bias})_basis_{Path(self.basis_model_dict["model_name"]).stem}.pkl'

    def _prepare_probs_for_oscr(self, logits, targets):
        all_gt = targets
        all_probs = torch.nn.functional.softmax(logits, dim=1)
        # clip probability scores of dummy class from the probability tensor and only work with the
        # probability scores of the known classes. This ensures compatibility with the OSCR definition
        all_probs = all_probs[:, :-1]

        # replace target of unkown unknown classes with target of known unkown classes as the
        # default implementation of the VAST library method get_known_unknown_indx() works with -1
        all_gt = np.array(all_gt)
        all_gt[all_gt == self.unknown_unknown_target] = self.known_unknown_target
        assert all_gt.min() == -1, 'target for KUCs and UUCs must be -1'
        all_prob, all_predicted = torch.max(all_probs, dim=1)

        return torch.tensor(all_gt), all_prob, all_predicted

    def _save_model(self, eval_metric, model_name, state_dict, bias):
        obj_serializable = {'approach_train': type(self).__name__, 'model_name': model_name, 'instance': {
            'protocol': self.protocol, 'gpu': self.gpu, 'ku_target': self.known_unknown_target, 'uu_target': self.unknown_unknown_target, 'model_path': self.model_path, 'log_path': self.log_path, 'oscr_path': self.oscr_path, 'train_cls': self.train_classes, 'architecture': self.architecture, 'epochs_finetune': self.epochs_finetune, 'basis_model_dict': self.basis_model_dict, 'lambda1': self.lambda1, 'lambda2': self.lambda2, 'alpha': self.alpha, 'no_dummy_clfs': self.no_dummy_clfs, 'compute_bias': self.compute_bias}, 'epoch_opt': self.epoch_current, 'eval_metric': eval_metric, 'eval_metric_opt': self.eval_metric_best, 'state_dict': state_dict, 'bias': bias}
        torch.save(obj_serializable, self.model_path / model_name)

    def _set_seeds(self, device):
        # modification msuter: move seed configuration logic from finetune_proser() into the Proser class
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUDA_VISIBLE_DEVICES'] = device.type
        torch.manual_seed(0)
        if os.environ['CUDA_VISIBLE_DEVICES']:
            torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        if os.environ['CUDA_VISIBLE_DEVICES']:
            torch.cuda.manual_seed(0)

    def _validate(self, net, val_loader, bias):
        logits_all, targets_all = self._extract(net, val_loader, bias)

        with torch.no_grad():
            auc, auc_deltaP = os_alg.valdummy(
                val_logits=logits_all, val_targets=targets_all)
            self.validation_metric_epochs['auc'].append(auc)
            self.validation_metric_epochs['auc_deltaP'].append(auc_deltaP)

            conf_knowns, conf_vast_unknowns, conf_dummy_unknowns, no_knowns, no_unknowns = os_alg.getConfidence(
                logits_all, targets_all, unknown_target=self.known_unknown_target)
            confidence_vast = torch.mean(
                torch.tensor([(conf_knowns/no_knowns).tolist(), (conf_vast_unknowns/no_unknowns).tolist()]))
            confidence_dummy = torch.mean(
                torch.tensor([(conf_knowns/no_knowns).tolist(), (conf_dummy_unknowns/no_unknowns).tolist()]))

            self.validation_metric_epochs['confidence_knowns'].append(
                conf_knowns.tolist())
            self.validation_metric_epochs['confidence_vast_unknowns'].append(
                conf_vast_unknowns.tolist())
            self.validation_metric_epochs['confidence_dummy_unknowns'].append(
                conf_dummy_unknowns.tolist())
            self.validation_metric_epochs['no_knowns_unknowns'].append(
                (no_knowns.tolist(), no_unknowns.tolist()))
            self.validation_metric_epochs['confidence_vast'].append(
                confidence_vast)
            self.validation_metric_epochs['confidence_dummy'].append(
                confidence_dummy)

            if auc_deltaP > self.eval_metric_best or self.epoch_current == 1:
                self.eval_metric_best = auc_deltaP
                self.epoch_best = self.epoch_current
                model_name = self._get_model_name()
                self._save_model('auc_deltaP',
                                 model_name, net.state_dict(), bias)
                self.model_path_best = model_name
