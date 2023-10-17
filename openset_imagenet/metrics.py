""" Various functions to calculate confidence and AUC."""
import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as f  # It was torch.functional as f


def confidence(scores, target_labels, offset=0., unknown_class = -1, last_valid_class = None):
    """ Returns model's confidence, Taken from https://github.com/Vastlab/vast/tree/main/vast.

    Args:
        scores(tensor): Softmax scores of the samples.
        target_labels(tensor): Target label of the samples.
        offset(float): Confidence offset value, typically 1/number_of_classes.
        unknown_class(int) which index to consider as unknown
        last_valid_class(int or None) which classes to predict; can be None for all and -1 for BG approach

    Returns:
        kn_conf: Confidence of known samples.
        kn_count: Count of known samples.
        neg_conf: Confidence of negative samples.
        neg_count Count of negative samples.
    """
    with torch.no_grad():
        unknown = target_labels == unknown_class
        known = torch.logical_and(target_labels >= 0, ~unknown)
        kn_count = sum(known).item()    # Total known samples in data
        neg_count = sum(unknown).item()  # Total negative samples in data
        kn_conf = 0.0
        neg_conf = 0.0
        if kn_count:
            # Average confidence known samples
            kn_conf = torch.sum(scores[known, target_labels[known]]).item() / kn_count
        if neg_count:
            # we have negative labels in the validation set
            neg_conf = torch.sum(
                1.0
                + offset
                - torch.max(scores[unknown,:last_valid_class], dim=1)[0]
            ).item() / neg_count

    return kn_conf, kn_count, neg_conf, neg_count


def auc_score_binary(target_labels, pred_scores, unk_label=-1):
    """ Calculates the binary AUC, all known samples labeled as 1. All negatives labeled as -1
    (or -2 if measuring unknowns).

    Args:
        target_labels(numpy array): Target label of the samples.
        pred_scores(numpy array): Softmax scores dim=[n_samples, n_classes] or
                                [n_samples, n_classes-1] if the loss is BGSoftmax
        unk_class(int): Class reserved for unknown samples.

    Returns:
        Binary AUC between known and unknown samples.
    """
    if torch.is_tensor(target_labels):
        target_labels = target_labels.cpu().detach().numpy()
    else:
        target_labels = target_labels.copy()
    if torch.is_tensor(pred_scores):
        pred_scores = pred_scores.cpu().detach().numpy()

    max_scores = np.max(pred_scores, axis=1)

    unknown = target_labels == unk_label
    known = target_labels >= 0
    used = np.logical_or(known, unknown)
    target_labels[known] = 1
    target_labels[unknown] = -1
#    breakpoint()
    return sklearn.metrics.roc_auc_score(target_labels[used], max_scores[used])


def auc_score_multiclass(target_labels, pred_scores):
    """ Calculates the multiclass AUC, each class against the rest.

    Args:
        target_labels(numpy array): Target label of the samples.
        pred_scores(numpy array): Predicted softmax scores of the samples.

    Returns:
        Multiclass AUC: measures the mean AUC including known and negatives.
    """
    if torch.is_tensor(target_labels):
        target_labels = target_labels.cpu().detach().numpy()
    if torch.is_tensor(pred_scores):
        pred_scores = pred_scores.cpu().detach().numpy()

    return sklearn.metrics.roc_auc_score(target_labels, pred_scores, multi_class="ovr")
