""" Code taken from the vast library https://github.com/Vastlab/vast"""
from vast import losses
from vast import tools
from torch.nn import functional as f
import torch


class ObjectoLoss:
    """ Simple wrapping class to handle losses, the class is only intended to keep code
    consistency.
    """
    def __init__(self, n_classes, unk_weight=1, norm_xi=10):
        # Entropic open-set term of the loss
        self.entropic = EntropicLoss(n_classes, unk_weight)
        # Norm penalty of the loss
        self.objecto = losses.objectoSphere_loss(knownsMinimumMag=norm_xi)
        self.entropic_value = None
        self.objecto_value = None

    def __call__(self, features, logits, targets, alpha, sample_weights=None):
        objecto_term = self.objecto(features, targets, sample_weights, reduction="sum")
        entropic_term = self.entropic(logits, targets, sample_weights)
        loss = entropic_term + alpha * objecto_term
        self.entropic_value = entropic_term.item()
        self.objecto_value = alpha * objecto_term .item()
        return loss


class EntropicLoss:
    """ Simple wrapping class to handle losses, the class is only intended to keep code
    consistency.
    """
    def __init__(self, n_classes, unk_weight=1):
        self.entropic = EntropicOpensetLoss(n_classes, unk_weight)

    def __call__(self, logits, targets, sample_weights=None):
        return self.entropic(logits, targets, sample_weights, reduction="sum")


class EntropicOpensetLoss:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes=10, unk_weight=1):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.ones = tools.device(torch.ones(self.class_count))
        self.unknowns_multiplier = unk_weight / self.class_count

    @tools.loss_reducer
    def __call__(self, logits, target, sample_weights=None):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        kn_idx = target != -1
        unk_idx = ~kn_idx
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[target[kn_idx]]

        categorical_targets[unk_idx, :] = (
            self.ones.expand(
                torch.sum(unk_idx).item(), self.class_count
            ) * self.unknowns_multiplier
        )

        log_values = f.log_softmax(logits, dim=1)
        negative_log_values = -1 * log_values
        loss = negative_log_values * categorical_targets
        sample_loss = torch.sum(loss, dim=1)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss


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
        return f"{self.avg:.2e}"


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
