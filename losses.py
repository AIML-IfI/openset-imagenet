# Code taken from the vast library https://github.com/Vastlab/vast
from vast import losses
from vast import tools
from torch.nn import functional as F
import torch


# Simple wrapping class to handle losses, the class is only intended to keep code consistency.
class objecto_loss():
    def __init__(self, n_classes, unk_weight=1, xi=10):
        """

        Args:
            n_classes:
            unk_weight:
            xi:
        """
        self.entropic = entropic_loss(n_classes, unk_weight)    # Entropic term of the loss
        self.objecto = losses.objectoSphere_loss(knownsMinimumMag=xi) # Objectosphere term

    def __call__(self, features, logits, targets, alpha, sample_weights=None):
        """

        Args:
            features:
            logits:
            targets:
            alpha:
            sample_weights:

        Returns:

        """
        objecto_term = self.objecto(features, targets, sample_weights, reduction='sum')
        entropic_term = self.entropic(logits, targets, sample_weights)
        loss = entropic_term + alpha*objecto_term
        self.entropic_value = entropic_term.item()
        self.objecto_value = alpha*objecto_term .item()
        return loss


# Simple wrapping class to handle losses
class entropic_loss():
    def __init__(self, n_classes, unk_weight=1):
        self.entropic = entropic_openset_loss(n_classes, unk_weight)

    def __call__(self, logits, targets, sample_weights=None):
        return self.entropic(logits, targets, sample_weights, reduction='sum')
        # print('loss inside', l.item())
        # return l


# Taken from vast, modified to accept mini batches without positive examples.
class entropic_openset_loss:
    def __init__(self, num_of_classes=10, unk_weight=1):
        self.num_of_classes = num_of_classes
        self.eye = tools.device(torch.eye(self.num_of_classes))
        self.ones = tools.device(torch.ones(self.num_of_classes))
        self.unknowns_multiplier = unk_weight / self.num_of_classes

    @tools.loss_reducer
    def __call__(self, logit_values, target, sample_weights=None):
        categorical_targets = tools.device(torch.zeros(logit_values.shape))
        known_indexes = target != -1
        unknown_indexes = ~known_indexes
        if torch.any(known_indexes):  # check if there is known samples in the batch
            categorical_targets[known_indexes, :] = self.eye[target[known_indexes]]
        categorical_targets[unknown_indexes, :] = (
            self.ones.expand((torch.sum(unknown_indexes).item(), self.num_of_classes))
            * self.unknowns_multiplier
        )
        log_values = F.log_softmax(logit_values, dim=1)
        negative_log_values = -1 * log_values
        loss = negative_log_values * categorical_targets
        sample_loss = torch.sum(loss, dim=1)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss


# Taken from https://github.com/pytorch/examples/tree/master/imagenet
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.2e}'.format(self.avg)


# Taken from https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks/
class EarlyStopping:
    """
    Early stops the training if validation loss/metrics doesn't improve after a given patience
    """
    def __init__(self, patience=100, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 100
            verbose (bool): If True, prints a message for each validation loss improvement. Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
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