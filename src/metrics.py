import torch
import torch.nn.functional as f  # TODO: it was torch.functional as f
from sklearn import metrics
import numpy as np


def confidence(scores, target_labels, offset=0.1):
    """ Returns model's confidence, Taken from https://github.com/Vastlab/vast/tree/main/vast.

    Args:
        scores: Softmax scores of the samples.
        target_labels: Target label of the samples.
        offset: Confidence offset value, typically 1/number_of_classes.

    Returns:
        kn_conf: Confidence of known samples.
        kn_count: Count of known samples.
        neg_conf: Confidence of negative samples.
        neg_count Count of negative samples.
    """
    with torch.no_grad():
        known = target_labels >= 0
        kn_count = sum(known).item()    # Total known samples in data
        neg_count = sum(~known).item()  # Total negative samples in data
        kn_conf = 0.0
        neg_conf = 0.0
        if kn_count:
            # Average confidence known samples
            kn_conf = torch.sum(scores[known, target_labels[known]]).item() / kn_count
        if neg_count:
            # Average confidence unknown samples
            neg_conf = torch.sum(1.0 + offset - torch.max(scores[~known], dim=1)[0]).item() / neg_count
    return kn_conf, kn_count, neg_conf, neg_count


def predict_objectosphere(logits, features, threshold):
    """ Predicts the class and softmax score of the input samples. Uses the product norms*score to threshold
    the unknown samples.

    Args:
        logits: Logit values of the samples.
        features: Deep features of the samples.
        threshold: Threshold value to discard unknowns.

    Returns:
        Tensor of predicted classes and predicted score.
    """
    scores = f.softmax(logits, dim=1)
    pred_score, pred_class = torch.max(scores, dim=1)
    norms = torch.norm(features, p=2, dim=1)
    unk = (norms*pred_score) < threshold
    pred_class[unk] = -1
    return torch.stack((pred_class, pred_score), dim=1)


def auc_score_binary(target_labels, pred_scores, unk_class=-1):
    """ Calculates the binary AUC, all known samples labeled as 1. All negatives labeled as -1
    (or -2 if measuring unknowns).

    Args:
        target_labels: Target label of the samples.
        pred_scores: Softmax scores dim = [n_samples, n_classes] or [n_samples, n_classes-1] if BGSoftmax
        unk_class: Class reserved for unknown samples.

    Returns:
        Binary AUC between known and unknown samples.
    """
    if torch.is_tensor(target_labels):
        target_labels = target_labels.cpu().detach().numpy()
    if torch.is_tensor(pred_scores):
        pred_scores = pred_scores.cpu().detach().numpy()

    max_scores = np.max(pred_scores, axis=1)

    kn = target_labels != unk_class
    target_labels[kn] = 1
    target_labels[~kn] = -1
    return metrics.roc_auc_score(target_labels, max_scores)


def auc_score_multiclass(target_labels, pred_scores):
    """ Calculates the multiclass AUC, each class against the rest.

    Args:
        target_labels: Target label of the samples.
        pred_scores: Predicted softmax scores of the samples.

    Returns:
        Multiclass AUC: measures the mean AUC including known and negatives.
    """
    if torch.is_tensor(target_labels):
        target_labels = target_labels.cpu().detach().numpy()
    if torch.is_tensor(pred_scores):
        pred_scores = pred_scores.cpu().detach().numpy()

    return metrics.roc_auc_score(target_labels, pred_scores, multi_class='ovr')
