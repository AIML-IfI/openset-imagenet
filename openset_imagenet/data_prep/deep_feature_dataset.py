import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import torch
from tqdm import tqdm
import csv
# instantiate module logger
logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self, trained_dnn, gpu, approach):
        """
        Args:
            trained_dnn (torch.nn.Module or a sublclass): trained DNN that returns a tuple in form of (logits, deep_features)
            gpu (string): specification of GPU to be used on the server
            approach (string): specification of the approach. Must be in {EVM, OpenMax}
        """
        self.trained_dnn = trained_dnn
        self.device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
        self.approach = approach
        self.df_dim = None
        self.kkc = None

    def extract_train_features(self, train_loader):
        """
        Args:
            train_loader (torch.utils.data.DataLoader): dataloader containing training samples on which deep features are extracted

        Returns:
            dict: dictionary with K representing class targets and V given by the extracted, class-specific (deep) features
        """
        targets, features, logits = self._extract(train_loader, 'training')

        # storing and returning the KKCs ensures that EVM can use KUCs for computing the pairwise
        # sample distances in EVM_Training() while only building a model for every KKCs
        self.kkc = self._collect_pos_classes(targets)
        print(self.kkc, logits.shape[-1])
        assert len(
            self.kkc) == logits.shape[-1], 'missing KKCs in provided training dataset. Check if original protocols are used'

        targets, features, logits = self._postprocess_train_data(
            targets, features, logits)
        # note: the postprocessed dataset needs to have the same number of KKCs. Otherwise, OpenMax
        # does not perform as expected
        pos_classes = self._collect_pos_classes(targets)
        print(len(pos_classes), self.kkc, logits.shape[-1])
        assert len(
            pos_classes) == logits.shape[-1], 'missing KKCs in post-processed dataset. Ensure that underlying, extracting DNN shows a sufficient closed set performance'

        feat_dict, _ = self._compose_dicts(targets, features, logits)
        return feat_dict, self.kkc

    def extract_features_and_logits(self, data_loader):
        """
        Args:
            data_loader (torch.utils.data.DataLoader): dataloader containing samples on which deep features/logits are extracted

        Returns:
            dict: dictionary with K representing class targets and V given by the extracted, class-specific (deep) features
            dict: dictionary with K representing class targets and V given by the extracted, class-specific logits
        """
        targets, features, logits = self._extract(data_loader)
        feat_dict, logit_dict = self._compose_dicts(targets, features, logits)
        return feat_dict, logit_dict

    def store_data_dict(self, data_dict, file_path_out, pos_classes=None):
        """
        Args:
            data_dict (dict): dictionary with K representing class targets and V given by the extracted, class-specific (deep) features or logits
            file_path_out (string, pathlib.Path): full filesystem path of the to be stored dict file (excluding the file extension)
            pos_classes (list<int>): list of positive classes used as a training argument. Hence, only provide together with the training feat dict
        """
        file_handler = open(str(Path(file_path_out)) + '.pkl', 'wb')
        obj_serializable = {'pos_classes': pos_classes, 'data': data_dict}
        pickle.dump(obj_serializable, file_handler)

    def load_data_dict(self, data_dict_path):
        """
        Args:
            data_dict_path (pathlib.Path, string): full filesystem path to the file that stores the serialized data dict
        """
        file_handler = open(data_dict_path, 'rb')
        data_dict = pickle.load(file_handler)
        logger.debug('\n')
        logger.info(f'Loaded data dict: {Path(data_dict_path).stem}')
        logger.info(
            f'Number of samples in the loaded data dict: {sum([data_dict["data"][key].shape[0] for key in data_dict["data"].keys()])}')
        logger.info(
            f'Number of classes in the loaded data dict (i.e. # dict keys): {len(list(data_dict["data"].keys()))}')

        return data_dict

    def _collect_pos_classes(self, targets):
        targets_unique = torch.unique(targets, sorted=True)
        pos_classes = targets_unique[targets_unique >=
                                     0].numpy().astype(np.int32).tolist()
        return pos_classes

    def _compose_dicts(self, targets, features, logits):

        df_data = pd.DataFrame(torch.hstack((targets, features, logits)).numpy(), columns=[
                               'gt'] + [f'feat_{i+1}' for i in range(self.df_dim)] + [f'log_{j+1}' for j in range(logits.shape[-1])])
        df_data['gt'] = df_data['gt'].astype(np.int32)

        df_group = df_data.groupby('gt')
        feat_dict = (df_group.apply(lambda x: list(
            map(list, zip(*[x[f'feat_{i+1}'] for i in range(self.df_dim)])))).to_dict())
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
        logger.info(
            f'Number of samples included in the dict: {count_feat}')
        logger.info(
            f'Number of classes (i.e. # dict keys): {len(list(feat_dict.keys()))}')

        return feat_dict, logit_dict

    def _extract(self, data_loader, procedure='validation/test'):
        targets = []
        features = []
        logits = []

        with torch.no_grad():
            self.trained_dnn.to(self.device)
            self.trained_dnn.eval()
            logger.debug('\n')
            logger.info(
                f'{self.approach}: Extracting info and preparing {procedure} dataset:')
            logger.debug(
                '################################################################################')
            for x, t in tqdm(data_loader):
                x = x.to(self.device)

                log, feat = self.trained_dnn(x)

                if self.df_dim is None:
                    self.df_dim = feat.shape[-1] if len(feat.shape) > 1 else 1

                targets.extend(t.tolist())
                features.extend(feat.tolist())
                logits.extend(log.tolist())

        targets, features, logits = torch.Tensor(
            targets)[:, None], torch.Tensor(features), torch.Tensor(logits)

        num_classes = len(self._collect_pos_classes(targets))
        assert num_classes == logits.shape[-1], 'missing KKCs in provided dataset. Check if original protocols are used'

        logger.debug('\n')
        logger.info(f'Number of extracted samples: {targets.shape[0]}')
        logger.info(f'Number of unique classes: {num_classes}')
        return targets, features, logits

    def _postprocess_train_data(self, targets, features, logits):
        pass


class EVMExtractor(Extractor):
    def __init__(self, trained_dnn, gpu, approach):
        super().__init__(trained_dnn, gpu, approach)

    def extract_features_and_logits(self, data_loader):
        """
        Args:
            data_loader (torch.utils.data.DataLoader): dataloader containing samples on which deep features/logits are extracted

        Returns: 
            dict: dictionary with K representing class targets and V given by the extracted, class-specific (deep) features
            None (since EVM does not need a dict containing logits)
        """
        feat_dict, _ = super().extract_features_and_logits(data_loader)
        return feat_dict, None

    def _postprocess_train_data(self, targets, features, logits):
        """
        Note: EVM uses all samples of the training set to train its model.
        """
        return targets, features, logits


class OpenMaxExtractor(Extractor):
    def __init__(self, trained_dnn, gpu, approach):
        super().__init__(trained_dnn, gpu, approach)

    def extract_features_and_logits(self, data_loader):
        """
        Args:
            data_loader (torch.utils.data.DataLoader): dataloader containing samples on which deep features/logits are extracted

        Returns: 
            dict: dictionary with K representing class targets and V given by the extracted, class-specific (deep) features
            dict: dictionary with K representing class targets and V given by the extracted, class-specific logits
        """
        feat_dict, logit_dict = super().extract_features_and_logits(data_loader)
        return feat_dict, logit_dict

    def _postprocess_train_data(self, targets, features, logits):
        # Note: OpenMax uses only the training samples that get correctly classified by the
        # underlying, extracting DNN to train its model.
        logger.debug('\n')
        logger.info(f'{self.approach} post-processing:')

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
                f'Number of unique classes after post-processing: {len(self._collect_pos_classes(targets_kkc[correct_idxs]))}')
            
            rows = zip(correct_idxs, targets_kkc[correct_idxs])
            with open('targets.txt', 'w') as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
            
            return targets_kkc[correct_idxs], features_kkc[correct_idxs], logits_kkc[correct_idxs]
