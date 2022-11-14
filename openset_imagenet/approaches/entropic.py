from . import approach_hierarchy
from .. import eval
import logging
import random
import torch
from tqdm import tqdm
import vast


random.seed(0)
# TODO: set torch seeds for reproducibility

# instantiate module logger
logger = logging.getLogger(__name__)


class EntropicOSL(approach_hierarchy.TrainDNNApproach):
    def __init__(self, protocol, gpu, ku_target, uu_target, model_path, log_path, oscr_path, train_cls, architecture, df_dim, num_cls, optimizer, epochs=200):
        super().__init__(protocol, gpu, ku_target, uu_target, model_path,
                         log_path, oscr_path, train_cls, architecture, df_dim, num_cls, optimizer, epochs)

    def train(self, train_loader, val_loader):
        logger.debug('\n')
        logger.info(f'Used loss: {vast.losses.entropic_openset_loss}')
        super().train(train_loader, val_loader, vast.losses.entropic_openset_loss(
            num_of_classes=len(self.known_targets)))

    def _validate(self, val_loader):

        weighted_confidences = torch.zeros(4, dtype=float)

        with torch.no_grad():
            self.model.eval()
            for x, t in tqdm(val_loader):
                x, t = x.to(self.device), t.to(self.device)

                logits, _ = self.model(x)
                assert self.known_unknown_target == -1

                weighted_confidences += torch.tensor(eval.split_confidence(
                    logits, t, negative_offset=(1.0/len(self.known_targets)), unknown_class=self.known_unknown_target))

            logger.info(
                f'conf_knowns: {weighted_confidences[0]} \t no_knowns: {weighted_confidences[2]}')
            logger.info(
                f'conf_unknowns: {weighted_confidences[1]} \t no_unknowns: {weighted_confidences[3]}')

            avg_conf_knowns = weighted_confidences[0]/weighted_confidences[2]
            avg_conf_unknowns = weighted_confidences[1]/weighted_confidences[3]
            weighted_avg_conf = (avg_conf_knowns + avg_conf_unknowns) / 2.0

            self.validation_metric_epochs['avg_conf_knowns'].append(
                avg_conf_knowns.item())
            self.validation_metric_epochs['avg_conf_unknowns'].append(
                avg_conf_unknowns.item())
            self.validation_metric_epochs['weighted_avg_conf'].append(
                weighted_avg_conf.item())

        logger.info(
            f'Validation Metric (VAST):\t Avg. Confidence on Knowns: {avg_conf_knowns*100:.4f} \t Avg. Confidence on Unknowns: {avg_conf_unknowns*100:.4f}\t Weighted Avg. Confidence: {weighted_avg_conf*100:.4f}')

        if weighted_avg_conf > self.eval_metric_best or self.epoch_current == 1:
            self.eval_metric_best = weighted_avg_conf
            self.epoch_best = self.epoch_current
            model_name = self._get_model_name()
            self._save_model('weighted_avg_conf', model_name)

            return self.model_path / model_name
