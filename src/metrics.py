import torch
import torch.functional as f
from sklearn import metrics


def confidence(scores, target, offset=0.1):
    """
    Returns model's confidence. Taken from https://github.com/Vastlab/vast/tree/main/vast
    Args:
        scores: Softmax scores of the samples.
        target: Target label of the samples.
        offset: Confidence offset value, typically 1/number_of_classes.
    Returns: Confidence value.
    """
    with torch.no_grad():
        known = target >= 0
        len_kn = sum(known).item()   # Total known samples in data
        len_un = sum(~known).item()    # Total unknown samples in data
        conf_kn = 0.0
        conf_un = 0.0
        if len_kn:
            # Average confidence known samples
            conf_kn = torch.sum(scores[known, target[known]]).item() / len_kn
        if len_un:
            # Average confidence unknown samples
            conf_un = torch.sum(1.0 + offset - torch.max(scores[~known], dim=1)[0]).item() / len_un
    return conf_kn, len_kn, conf_un, len_un


def predict_objectosphere(logits, features, threshold):
    """
    Predicts the class and softmax score of the input samples. Uses the product norms*score to threshold
    the unknown samples.
    Args:
        logits: Predicted logit values.
        features: Deep features of the samples.
        threshold: Threshold value to discard unknowns.
    Returns: Predicted classes
    """
    scores = f.softmax(logits, dim=1)
    pred_score, pred_class = torch.max(scores, dim=1)
    norms = torch.norm(features, p=2, dim=1)
    unk = (norms*pred_score) < threshold
    pred_class[unk] = -1
    return torch.stack((pred_class, pred_score), dim=1)


def auc_score_binary(t_true, pred_score, unk_class=-1):
    """ Calculates the binary AUC; known samples labeled as 1, known-unknown labeled as -1.
    Args:
        t_true: Target label of the samples.
        pred_score: Maximum predicted scores for a known class.
        unk_class: Class reserved for unknown samples.
    Returns: Binary AUC, measures the separation between known and unknown samples.
    """
    if torch.is_tensor(t_true):
        t_true = t_true.cpu().detach().numpy()
    if torch.is_tensor(pred_score):
        pred_score = pred_score.cpu().detach().numpy()

    kn = t_true != unk_class
    t_true[kn] = 1
    t_true[~kn] = -1
    return metrics.roc_auc_score(t_true, pred_score)


def auc_score_multiclass(t_true, pred_score):
    """
    Calculates the multiclass AUC; Calculates the AUC of each class against the rest.
    Args:
        t_true: Target label of the samples.
        pred_score: Predicted softmax score of the samples.

    Returns: Multiclass AUC, measures the mean AUC including known and known-unknowns.

    """
    if torch.is_tensor(t_true):
        t_true = t_true.cpu().detach().numpy()
    if torch.is_tensor(pred_score):
        pred_score = pred_score.cpu().detach().numpy()
    return metrics.roc_auc_score(t_true, pred_score, multi_class='ovr')
