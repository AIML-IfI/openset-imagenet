import torch
import torch.functional as f
from sklearn import metrics


def confidence(scores, target, negative_offset=0.1):
    """
    Returns model's confidence. Taken from https://github.com/Vastlab/vast/tree/main/vast
    Args:
        scores: Softmax scores of the samples.
        target: Target label of the samples.
        negative_offset: Confidence offset value, typically 1/number_of_classes.

    Returns: Confidence value.
    """
    with torch.no_grad():
        known = target >= 0
        conf = 0.0
        if torch.sum(known):
            conf += torch.sum(scores[known, target[known]])
        if torch.sum(~known):
            conf += torch.sum(1.0 + negative_offset - torch.max(scores[~known], dim=1)[0])
    return torch.tensor((conf, len(scores)))


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


def auc_score_binary(t_true, pred_score):
    """
    Calculates the binary AUC; known samples labeled as 1, known-unknown labeled as -1.
    Args:
        t_true: Target label of the samples.
        pred_score: Predicted softmax score of the samples.

    Returns: Binary AUC, measures the separation between known and unknown samples.
    """
    if torch.is_tensor(t_true):
        t_true = t_true.cpu().detach().numpy()
    if torch.is_tensor(pred_score):
        pred_score = pred_score.cpu().detach().numpy()

    t_true[(t_true > -1)] = 1
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
