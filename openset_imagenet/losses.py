""" Code taken from the vast library https://github.com/Vastlab/vast"""
from torch.nn import functional as f
import torch
from vast import tools


class EntropicOpensetLoss:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, unk_weight=1):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.unknowns_multiplier = unk_weight / self.class_count
        self.ones = tools.device(torch.ones(self.class_count)) * self.unknowns_multiplier
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, target):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = target < 0
        kn_idx = ~unk_idx
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        categorical_targets[unk_idx, :] = (
            self.ones.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )
        return self.cross_entropy(logits, categorical_targets)


class AverageMeter(object):
    """ Computes and stores the average and current value. Taken from
    https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.val, self.avg, self.sum, self.count = None, None, None, None
        self.reset()

    def reset(self):
        """ Sets all values to 0. """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        """ Update metric values.

        Args:
            val (flat): Current value.
            count (int): Number of samples represented by val. Defaults to 1.
        """
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.avg:3.3f}"


# Taken from:
# https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks/
class EarlyStopping:
    """ Stops the training if validation loss/metrics doesn't improve after a given patience"""
    def __init__(self, patience=100, delta=0):
        """
        Args:
            patience(int): How long wait after last time validation loss improved. Default: 100
            delta(float): Minimum change in the monitored quantity to qualify as an improvement
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        if loss is True:
            score = -metrics
        else:
            score = metrics

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
