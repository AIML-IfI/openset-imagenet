import torch
from sklearn import metrics


# taken from https://github.com/Vastlab/vast/tree/main/vast
# TODO: Question about target in confidence.
def confidence(scores, target, negative_offset=0.1):
    """Returns sums of confideces."""
    with torch.no_grad():
        known = target >= 0
        # pred = torch.nn.functional.softmax(logits, dim=1)
        #    import ipdb; ipdb.set_trace()
        confidence = 0.0
        if torch.sum(known):
            confidence += torch.sum(scores[known, target[known]])
        if torch.sum(~known):
            confidence += torch.sum(1.0 + negative_offset - torch.max(scores[~known], dim=1)[0])
    # return torch.tensor((confidence, len(logits)))
    return torch.tensor((confidence, len(scores)))

# def predict_ver2(targets, logits, threshold, loss, features=None):
#     """Returns a tensor with the predicted class and its confidence score, if features are given
#     predicts using score*norm(features) >= threshold, otherwie score >= threshold"""
#     with torch.no_grad():
#         scores = torch.nn.functional.softmax(logits, dim=1)
#         pred_score, pred_class = torch.max(scores, dim=1)
#         known = (targets != -1)
#         unknown = ~known

#         if features is not None:    # If features are provided we can use norm(features)*Softmax(x)
#             norms = torch.norm(features, p=2, dim=1)
#             pred_unk = norms*pred_score < threshold
#         else:
#             pred_unk = pred_score < threshold

#         pred_class[pred_unk] = -1
#         pred_score[pred_unk] = 1 - pred_score[unknown] + 1/logits.shape[1]

#     return pred_class, pred_score


# TODO: the threshold is norm(features)*Softmax(x). In validation we know if x is known
# So we use the target to correctly report the confidence of a sample.
# In test we dont know anything about x, the confidence is reported over the network prediction.
# why don't do the same in valudation
# def predict_objectosphere_val(targets, logits, features, threshold):
#     """Returns a tensor with the predicted class and its confidence"""
#     with torch.no_grad():
#         scores = torch.nn.functional.softmax(logits, dim=1)
#         pred_score, pred_class = torch.max(scores, dim=1)
#         norms = torch.norm(features, p=2, dim=1)
#         # According to Dhamija et al, theshold is against |feature|*softmax_score
#         known = (targets != -1)  # using knwledge of the targets to calculate confidence
#         # All indices that don't pass threshold are unknown.
#         unknown = norms*pred_score < threshold

#         # confidence: known - >max(S(x)), unknown -> 1 - max(S(x)) + 1/\c\
#         pred_class[unknown] = -1
#         pred_score[~known] = 1 - pred_score[~known] + 1/(logits.shape[1])
#     # return torch.stack((pred_class, pred_score), dim=1)
#     return pred_class, pred_score


# def predict_objectosphere(logits, features, threshold):
#     scores = torch.nn.functional.softmax(logits, dim=1)
#     pred_score, pred_class = torch.max(scores, dim=1)
#     norms = torch.norm(features, p=2, dim=1)
#     unk = (norms*pred_score) < threshold
#     pred_class[unk] = -1
#     return torch.stack((pred_class, pred_score), dim=1)
#     # return pred_class, pred_score


def auc_score_binary(t_true, pred_score):
    """AUC score for binary case. knowns to 1, unknowns stay -1."""
    if torch.is_tensor(t_true):
        t_true = t_true.cpu().detach().numpy()
    if torch.is_tensor(pred_score):
        pred_score = pred_score.cpu().detach().numpy()

    t_true[(t_true > -1)] = 1
    return metrics.roc_auc_score(t_true, pred_score)


def auc_score_multiclass(t_true, pred_score):
    if torch.is_tensor(t_true):
        t_true = t_true.cpu().detach().numpy()
    if torch.is_tensor(pred_score):
        pred_score = pred_score.cpu().detach().numpy()
    return metrics.roc_auc_score(t_true, pred_score, multi_class='ovr')
