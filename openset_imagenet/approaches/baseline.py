from . import approach_hierarchy
import logging
import random
import torch
from tqdm import tqdm
import vast

random.seed(0)

# instantiate module logger
logger = logging.getLogger(__name__)


class Baseline(approach_hierarchy.TrainDNNApproach):
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture, df_dim, num_cls, optimizer, epochs=0):
        super().__init__(protocol, gpu, ku_target, uu_target, model_path,
                         log_path, oscr_path, train_cls, architecture, df_dim, num_cls, optimizer, epochs)

    def train(self, train_loader, val_loader):
        logger.info(f'Used loss: {torch.nn.CrossEntropyLoss()}')
        super().train(train_loader, val_loader, torch.nn.CrossEntropyLoss())

    def _validate(self, val_loader):

        accuracy = torch.zeros(2, dtype=float)

        with torch.no_grad():
            self.model.eval()
            for x, t in tqdm(val_loader):
                x, t = x.to(self.device), t.to(self.device)

                z, _ = self.model(x)
                accuracy += vast.losses.accuracy(z, t)

        self.validation_metric_epochs['accuracy_val'].append(
            ((accuracy[0]/accuracy[1])*100).item())
        logger.info(
            f'Validation Accuracy:\t {(accuracy[0]/accuracy[1])*100:.4f}\t {accuracy[0]}/{accuracy[1]}')

        if (accuracy[0]/accuracy[1]) > self.eval_metric_best or self.epoch_current == 1:
            self.eval_metric_best = (accuracy[0]/accuracy[1].item())
            self.epoch_best = self.epoch_current
            model_name = self._get_model_name()
            self._save_model('accuracy', model_name)

            return self.model_path / model_name
