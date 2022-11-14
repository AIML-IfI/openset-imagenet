from .. import architectures, tools
from collections import defaultdict
import csv
import logging
import numpy as np
from pathlib import Path
import pickle
import time
import torch
from tqdm import tqdm
import random
import vast
from vast import opensetAlgos


random.seed(0)

# instantiate module logger
logger = logging.getLogger(__name__)


class RootApproach:
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture):
        """     
        Args:
            protocol (int): protocol on whose data the class runs, element of {1,2,3}
            gpu (string): specification of GPU to be used on the server
            ku_target (int): target that is assigned to known unkown classes in the dataset
            uu_target (int): target that is assigned to unkown unknown classes in the dataset
            model_path (string, pathlib.Path): filesystem path to directory where trained models are stored in
            log_path (string, pathlib.Path): filesystem path to directory where log data is stored in
            oscr_path (string, pathlib.Path): filesystem path to directory where files containing oscr metrics are stored in
            train_cls (List<str>): list of acronyms refering to the used classes in training
            architecture (string): architecture of the to be trained model / the underlying, feature extracting model      
        """
        self.protocol = protocol
        self.gpu = gpu
        self.device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

        self.known_unknown_target = ku_target
        self.unknown_unknown_target = uu_target

        self.model_path = model_path
        self.log_path = log_path
        self.oscr_path = oscr_path

        self.train_classes = train_cls
        self.architecture = architecture


class FeatureApproach(RootApproach):
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture, used_dnn, approach, fpr_thresholds):
        super().__init__(protocol, gpu, ku_target,
                         uu_target, model_path, log_path, oscr_path, train_cls, architecture)
        """
        Args:
            used_dnn (string): specifications of DNN that has been used to extract features from the datasets
            approach (string): specification of the approach used (i.e. element of {EVM, OpenMax})
            fpr_thresholds (list<float>): list of false positive rate (fpr) thresholds that are used to select the best model in the validation procedure in a "ccr@fpr" fashion (see Dhamija et al., 2018)
        """
        self.used_dnn = used_dnn
        self.approach = approach
        self.fpr_thresholds = fpr_thresholds

    def train(self, pos_classes, train_data, val_data, val_logits, hyperparams):
        """
        Args:
            pos_classes (int, list): list of classes being processed in current process, used for multi-process training procedure (see source code comments in https://github.com/Vastlab/vast/tree/main/vast/opensetAlgos/EVM.py for details)
            train_data (dict): dictionary with K representing class targets and V given by the class-specific features of the training set
            val_data (dict): dictionary with K representing class targets and V given by the class-specific features of the validation set
            val_logits (dict, None): dictionary with K representing class targets and V given by the class-specific logit vectors of the validation set or None (if logits are not needed for the approach)
            hyperparams (namedtuple): a named tuple containing the hyperparameter options to be used in the training procedure
        """

        logger.debug('\n')
        logger.info(f'Starting {self.approach} Training Procedure:')
        # Training method returns iterator over (hparam_combo, (class, {model}))
        training_fct = getattr(opensetAlgos, f'{self.approach}_Training')
        all_hyper_param_models = list(training_fct(
            pos_classes_to_process=pos_classes, features_all_classes=train_data, args=hyperparams, gpu=self.device.index, models=None))

        # integrating returned models in a data structure as required by <self.approach>_Inference()
        hparam_combo_to_model = defaultdict(list)

        for i in range(len(all_hyper_param_models)):
            hparam_combo_to_model[all_hyper_param_models[i][0]].append(
                all_hyper_param_models[i][1])
        logger.info(
            f'Trained models associated with hyperparameters: {list(hparam_combo_to_model.keys())}')

        for key in hparam_combo_to_model:
            hparam_combo_to_model[key] = dict(hparam_combo_to_model[key])

            # store models per hyperparameter combination as a (hparam_combo, model)-tuple
            model_name = f'p{self.protocol}_traincls({"+".join(self.train_classes)})_{self.approach.lower()}_{key}_{hyperparams.distance_metric}_dnn_{self.used_dnn}.pkl'

            file_handler = open(self.model_path / model_name, 'wb')
            obj_serializable = {'approach_train': type(self).__name__, 'model_name': model_name, 'hparam_combo': key, 'distance_metric': hyperparams.distance_metric, 'instance': {'protocol': self.protocol, 'gpu': self.gpu, 'ku_target': self.known_unknown_target, 'uu_target': self.unknown_unknown_target,
                                                                                                                                                                                   'model_path': self.model_path, 'log_path': self.log_path, 'oscr_path': self.oscr_path, 'train_cls': self.train_classes, 'architecture': self.architecture, 'used_dnn': self.used_dnn, 'fpr_thresholds': self.fpr_thresholds}, 'model':  hparam_combo_to_model[key]}
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

        for key in hparam_combo_to_model:
            logger.info(
                f'Starting {self.approach} Validation on Hyperparameter Combination: {key}')

            self._validate(list(val_data.keys()), val_data, val_logits, hyperparams,
                           key, hparam_combo_to_model[key])

    def _compute_probabilities(self, pos_classes_val, val_data, hyperparams, models):
        """
        Args:
            pos_classes_val(int, list): list of classes in the validation/test set being processed in current process, used for multi-processing (see source code comments in https://github.com/Vastlab/vast/tree/main/vast/opensetAlgos/EVM.py for details)
            val_data(dict): dictionary with K representing class targets and V given by the class-specific features of the validation/test set
            hyperparams(namedtuple): named tuple containing the hyperparameter options. Only the 'distance_metric' argument is actually used during inference
            models(dict): dictionary containing the model that is associated with a specific hyperparameter combination. K representing training class targets and V given by class-specific model as computed in training
        """

        """
        Important: Since the <approach>_Inference() function in the vast package sorts the 
        keys of the collated model, the semantic of the returned probabilities depends on 
        the type of the dictionary keys. For example, when sorting is applied on the 'stringified'
        integer classes, the column indices of the returned probabilities tensor do not necessarily
        correspond with the integer class targets. Hence, the assertion for integer type below. 
        
        """
        assert sum([isinstance(k, int) for k in models.keys()]) == len(
            list(models.keys())), 'dictionarys keys are not of type "int"'

        # conversion to float64 necessary to avoid runtime error by '/torch/functional.py'
        val_data = {k: v.double() for k, v in val_data.items()}

        probabilities = list(getattr(opensetAlgos, f'{self.approach}_Inference')(pos_classes_to_process=pos_classes_val,
                                                                                features_all_classes=val_data, args=hyperparams, gpu=self.device.index, models=models))

        return dict(list(zip(*probabilities))[1])

    def _get_ccr_at_fpr(self, hparam_combo, path, fpr, ccr):
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

        for e in self.fpr_thresholds:
            threshold = torch.Tensor([e]*fpr.shape[0])
            _, min_idx = torch.min(torch.abs((fpr - threshold)), -1)
            values.append((hparam_combo, e, min_idx.item(),
                          fpr[min_idx].item(), ccr[min_idx].item()))

        with open(path, 'w') as file_out:
            writer = csv.writer(file_out)
            for e in values:
                writer.writerow(e)

        return values

    def _get_filename_oscr(self, procedure, hparam_combination, distance_metric):
        return f'p{self.protocol}_{self.approach.lower()}_{procedure}_{hparam_combination}_{distance_metric}_oscr_values_dnn_{self.used_dnn}.csv'

    def _load_model(self, saved_model_path):
        """
        Args:
            saved_model_path (string): full filesystem path to the file that stores the best performing model in training
        """

        file_handler = open(saved_model_path, 'rb')
        model_dict = pickle.load(file_handler)
        logger.info(
            f'Loaded model associated with hyperparameter combination {model_dict["hparam_combo"]}')
        assert sum([isinstance(k, int) for k in model_dict['model'].keys()]) == len(
            list(model_dict['model'].keys())), 'dictionarys keys are not of type "int"'

        return model_dict

    def _prepare_probs_for_oscr(self, dict_probs):
        all_gt = []
        all_probs = []

        for key in dict_probs.keys():
            all_gt.extend([key]*dict_probs[key].shape[0])
            all_probs.extend(dict_probs[key].tolist())

        # replace target of unkown unknown classes with target of known unkown classes as the
        # default implementation of the library method get_known_unknown_indx() works with -1
        all_gt = np.array(all_gt)
        all_gt[all_gt == self.unknown_unknown_target] = self.known_unknown_target
        assert all_gt.min() == -1, 'target for KUCs and UUCs must be -1'
        all_prob, all_predicted = torch.max(torch.tensor(all_probs), dim=1)
        print("Preparing probabilities for OSCR, and shape for all_gt, and all_prob", all_gt.shape, all_prob.shape)
        return all_gt, all_prob, all_predicted


class TrainDNNApproach(RootApproach):
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture, df_dim, num_cls, optimizer, epochs):
        super().__init__(protocol, gpu, ku_target,
                         uu_target, model_path, log_path, oscr_path, train_cls, architecture)
        """
        Args:
            df_dim (int): number of dimensions of the deep feature space
            optimizer (dict): dictionary holding required info/parameters for instantiating the PyTorch optimizer
            epochs (int): number of epochs
            num_cls (int): number of output dimensions of the logit layer
        """
        vast.tools.set_device_gpu(self.device.index) if (
            self.device.type == 'cuda') else vast.tools.set_device_cpu()

        self.df_dim = df_dim
        self.optimizer = optimizer
        self.epochs = epochs
        self.epoch_current = None

        self.known_targets = tuple(range(0, num_cls))
        self.known_targets_onehot = {k: torch.eye(len(self.known_targets))[
            i] for i, k in enumerate(self.known_targets)}

        self.model = getattr(architectures, architecture)(
            feature_dim=df_dim, num_classes=num_cls)

        if self.optimizer['opt'] == 'Adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.optimizer['lr'], betas=(
                self.optimizer['beta_1'], self.optimizer['beta_2']))
        else:
            self.optim = torch.optim.SGD(self.model.parameters(
            ), lr=self.optimizer['lr'], momentum=self.optimizer['beta_1'])

        logger.info(
            f'Optimizer: {type(self.optim)} - Learning Rate: {self.optimizer["lr"]} - Beta_1: {self.optimizer["beta_1"]} - Beta_2: {self.optimizer["beta_2"] if list(self.optimizer.keys())[-1] == "beta_2" else None}')

        self.eval_metric_best = 0
        self.epoch_best = 0
        self.model_path_best = None
        self.training_metric_epochs = defaultdict(list)
        self.validation_metric_epochs = defaultdict(list)

    def train(self, train_loader, val_loader, loss_function):
        logger.debug('\n')
        logger.info(f'Starting {type(self).__name__} Training Procedure:')
        t0_training = time.time()
        self.model.to(self.device)

        criterion = loss_function

        tools.get_cuda_info(self.device)
        tools.get_model_status(self.model)

        for epoch in range(self.epochs):
            self.epoch_current = epoch + 1
            loss_history = []
            accuracy_train = torch.zeros(2, dtype=float)
            logger.debug('\n')
            logger.info(f'Training: Epoch {self.epoch_current}')
            t0_epoch = time.time()
            self.model.train()

            for x, t in tqdm(train_loader):
                x, t = x.to(self.device), t.to(self.device)
                self.optim.zero_grad()
                z, _ = self.model(x)
                J = criterion(z, t)
                loss_history.extend([float(J.tolist())]) if isinstance(
                    J.tolist(), list) == False else loss_history.extend(J.tolist())

                J.mean().backward()
                self.optim.step()
                accuracy_train += vast.losses.accuracy(z, t)

            self.training_metric_epochs['loss'].append(
                torch.mean(torch.tensor(loss_history)).item())
            self.training_metric_epochs['accuracy_train'].append(
                ((accuracy_train[0]/accuracy_train[1])*100).item())
            logger.info(
                f'Training time for epoch {self.epoch_current}:\t {((time.time() - t0_epoch) // 60):.0f}m {((time.time() - t0_epoch) % 60):.0f}s')
            logger.info(
                f'Average training loss:\t {torch.mean(torch.tensor(loss_history))}')
            logger.info(
                f'Training Accuracy:\t {(accuracy_train[0]/accuracy_train[1])*100:.4f}\t {accuracy_train[0]}/{accuracy_train[1]}')
            logger.debug('\n')
            logger.info(f'Validation: Epoch {self.epoch_current}')
            self.model_path_best = self._validate(val_loader)

        logger.info(
            f'Training time:\t {((time.time() - t0_training) // 60):.0f}m {((time.time() - t0_epoch) % 60):.0f}s')
        logger.info(
            f'Training Metric per Epoch: {self.training_metric_epochs}')
        logger.info(
            f'Validation Metric per Epoch: {self.validation_metric_epochs}')
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

        model = getattr(architectures, model_dict['instance']['architecture'])(
            model_dict['instance']['df_dim'], model_dict['instance']['num_cls'])
        model.load_state_dict(model_dict['state_dict'])
        model.to(self.device)

        all_gt = []
        all_prob = []

        with torch.no_grad():
            model.eval()
            for x, t in tqdm(test_loader):
                x, t = x.to(self.device), t.to(self.device)

                all_gt.extend(t.tolist())

                z, _ = model(x)
                all_prob.extend(
                    (torch.nn.functional.softmax(z, dim=1)).tolist())

        # replace target of unkown unknown classes with target of known unkown classes as the
        # default implementation of the library method get_known_unknown_indx() works with -1
        all_gt = np.array(all_gt)
        all_gt[all_gt == self.unknown_unknown_target] = self.known_unknown_target
        assert all_gt.min() == -1, 'target for KUCs and UUCs must be -1'
        all_prob, all_predicted = torch.max(torch.tensor(all_prob), dim=1)

        for i in range(len(all_predicted)):
            assert all_predicted[i].item(
            ) == self.known_targets[all_predicted[i].item()], 'Invalid target found in dataset'

        vast.tools.set_device_cpu()
        fpr, ccr, cover = vast.eval.tensor_OSRC(
            torch.tensor(all_gt), all_predicted, all_prob)

        tools.store_oscr_metrics(
            self.oscr_path / f'p{self.protocol}_{type(self).__name__.lower()}_test_{Path(model_dict["model_name"]).stem}_oscr_values.csv', fpr, ccr, cover)

    def _get_model_name(self):
        return f'p{self.protocol}_traincls({"+".join(self.train_classes)})_{type(self).__name__.lower()}_{self.architecture.lower()}_df{self.df_dim}_e{self.epochs}_opt{self.optimizer["opt"]}({"+".join([str(self.optimizer[key]) for key in self.optimizer.keys() if key != "opt"])}).pth'

    def _save_model(self, eval_metric, model_name):
        obj_serializable = {'approach_train': type(self).__name__, 'model_name': model_name, 'instance': {
            'protocol': self.protocol, 'gpu': self.gpu, 'ku_target': self.known_unknown_target, 'uu_target': self.unknown_unknown_target, 'model_path': self.model_path, 'log_path': self.log_path, 'oscr_path': self.oscr_path, 'train_cls': self.train_classes, 'architecture': self.architecture, 'df_dim': self.df_dim, 'num_cls': len(self.known_targets), 'optimizer': self.optimizer, 'epochs': self.epochs}, 'epoch_opt': self.epoch_current, 'eval_metric': eval_metric, 'eval_metric_opt': self.eval_metric_best, 'state_dict': self.model.state_dict()}
        torch.save(obj_serializable, self.model_path / model_name)

    def _validate(self, val_loader):
        pass
