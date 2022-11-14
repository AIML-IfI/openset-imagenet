import logging
#import settings
from .. import experiments
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
# modification msuter: delete unneeded matplotlib import statements

# instantiate module logger
logger = logging.getLogger(__name__)

"""
########################################################################################
 README
########################################################################################

This code is copied from https://gitlab.ifi.uzh.ch/aiml/projects/placeholder-open-set
and is associated with the project 'placeholder-open-set' by Omnia Elsaadany. The
project adapts the source code frome the following paper:

  Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan,
  Learning Placeholders for Open-Set Recognition,
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
  4401-4410, 2021
  Source Code: https://github.com/zhoudw-zdw/CVPR21-Proser

Detailed information on the project are available in the project description
(see Gitlab repo: https://gitlab.ifi.uzh.ch/aiml/projects/placeholder-open-set).

In order to make the code compatible with this Master Thesis, some adaptations are
necessary. Furthermore, some unneeded code blocks/functions are excluded. These
modifications by msuter are marked as such and are introduced by a comment line beginning
with 'modification msuter: ...'.

"""


# modification msuter: extend signature by argparse.Namespace object for accessing arguments
def dummypredict(net, x, args):
    """Takes inputs x and network net and returns dummy classifier max logit"""
    if args.backbone == "WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        # Return max logits for unknown classifiers
        out, _ = torch.max(out, dim=1)
        return out.view(-1, 1)

    # modification msuter: add logic for architectures used in experiments
    elif args.backbone in [arch + '_Feature' for arch in experiments.MODELS]:
        # apply forward pass of the respective model as well as the dummy classifiers
        _, features = net(x)
        out = net.clf2(features)
        # Return max logits for unknown classifiers
        out, _ = torch.max(out, dim=1)
        return out.view(-1, 1)

    # modification msuter: delete LeNet_plus_plus architecture


# modification msuter: extend signature by argparse.Namespace object for accessing arguments
def pre2block(net, x, args):
    """Takes inputs x and network net and returns the hidden representation (middle layer output), used in manifold mixup step"""
    if args.backbone == "WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        return out

    # modification msuter: add logic for architectures used in experiments
    elif args.backbone in [arch + '_Feature' for arch in experiments.MODELS]:
        """
        Note: for consistency reasons with the original source code of Zhou et al. (2021), the
        manifold mixup is performed after the third group of blocks (i.e. layer3). By following
        this approach, the manifold mixup is performed after the penultimate group/layer in both
        architectures (i.e WideResnet and ResNet)
        """
        out = net.conv1(x)
        out = net.bn1(out)
        out = net.relu(out)
        out = net.maxpool(out)

        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        return out

    # modification msuter: delete unneeded logic for LeNet_plus_plus architecture


# modification msuter: extend signature by argparse.Namespace object for accessing arguments
def latter2blockclf1(net, x, args):
    """Takes inputs x (hidden representation) and network net and returns logits for known classes, used in manifold mixup step"""
    if args.backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
        return out

    # modification msuter: add logic for architectures used in experiments
    elif args.backbone in [arch + '_Feature' for arch in experiments.MODELS]:
        out = net.layer4(x)
        out = net.avgpool(out)
        out = torch.flatten(out, 1)
        out_features = net.fc(out)
        out = net.fc2(out_features)
        return out

    # modification msuter: delete unneeded logic for LeNet_plus_plus architecture


# modification msuter: extend signature by argparse.Namespace object for accessing arguments
def latter2blockclf2(net, x, args):
    """Takes inputs x (hidden representation) and network net and returns dummy classifier max logit, used in manifold mixup step"""
    if args.backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        out, _ = torch.max(out, dim=1)
        return out.view(-1, 1)

    # modification msuter: add logic for architectures used in experiments
    elif args.backbone in [arch + '_Feature' for arch in experiments.MODELS]:
        out = net.layer4(x)
        out = net.avgpool(out)
        out = torch.flatten(out, 1)
        out_features = net.fc(out)
        out = net.clf2(out_features)
        out, _ = torch.max(out, dim=1)
        return out.view(-1, 1)

    # modification msuter: delete unneeded logic for LeNet_plus_plus architecture


# modification msuter: delete unneeded getmodel() function


# modification msuter: extend signature by argparse.Namespace object for accessing arguments, trainloader, and device
def traindummy(epoch, net, args, trainloader, device):
    criterion = nn.CrossEntropyLoss()
    # --lr argument is pretrained model learning rate, --lr used in finetuning is lr*0.1
    optimizer = optim.SGD(net.parameters(), lr=args.lr *
                          0.1, momentum=0.9, weight_decay=5e-4)

    # modification msuter: modify console info output
    logger.debug('\n')
    logger.info(f'Starting PROSER Finetuning: Epoch {epoch+1}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = args.alpha

    for idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        halflength = int(len(inputs)/2)
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()

        # Split the batch into two halves, 'prehalfinputs' would be used to generate the synthetic data(data placeholders)
        # while the 'laterhalfinputs' would be used as the known samples.
        prehalfinputs = inputs[:halflength]
        prehalflabels = targets[:halflength]
        laterhalfinputs = inputs[halflength:]
        laterhalflabels = targets[halflength:]

        # (eq. 6 in the paper) X~_pre
        index = torch.randperm(prehalfinputs.size(0)).to(device)
        pre2embeddings = pre2block(net, prehalfinputs, args)
        mixed_embeddings = beta * pre2embeddings + \
            (1 - beta) * pre2embeddings[index]

        # Masking the pairs of the same class,
        # passing only mixed embeddings created from different classes.
        mixed_embeddings = mixed_embeddings[prehalflabels !=
                                            prehalflabels[index]]

        # (eq. 7 in the paper) the input for l2
        # For placeholder data:
        # Get and concatentae logits for known and placeholder classes together.
        prehalfoutput = torch.cat((latter2blockclf1(
            net, mixed_embeddings, args), latter2blockclf2(net, mixed_embeddings, args)), 1)

        # For the 2nd half:
        # Get logits for dummy classifiers
        dummylogit = dummypredict(net, laterhalfinputs, args)
        # Get logits for known classes classifiers
        if args.backbone == 'LeNet_plus_plus':
            lateroutputs, _ = net(laterhalfinputs)
        # modification msuter: add logic for architectures used in experiments
        elif args.backbone in [arch + '_Feature' for arch in experiments.MODELS]:
            lateroutputs, _ = net(laterhalfinputs)
        else:
            lateroutputs = net(laterhalfinputs)
        # Concatentate all logit for both known and placeholder classes
        latterhalfoutput = torch.cat((lateroutputs, dummylogit), 1)

        # Setting the known class logit to -1e9 to force a classifier placeholder to be the second highest logit.
        # torch.cat((lateroutputs.clone(),dummylogit.clone()),dim=1)
        dummpyoutputs = latterhalfoutput.clone()
        for i in range(len(dummpyoutputs)):
            nowlabel = laterhalflabels[i]
            dummpyoutputs[i][nowlabel] = -1e9

        dummytargets = torch.ones_like(laterhalflabels)*args.known_class

        # Concatente all outputs from known samples and data placeholders
        outputs = torch.cat((prehalfoutput, latterhalfoutput), 0)
        # (eq. 7 in the paper), l2 forcing the placeholder data to be predicted as unknown (belongs to the dummy classifer).
        # Updated ones_like on target to work with variable size mixed embeddings (resulting from masking).
        prehalflabels = (torch.ones(prehalfoutput.size(dim=0))
                         * args.known_class).long().to(device)
        loss1 = criterion(prehalfoutput, prehalflabels)
        # (eq. 5 in the paper) 1st term in l1, forcing the target class classifier to include the known sample.
        loss2 = criterion(latterhalfoutput, laterhalflabels)
        # (eq. 5 in the paper) 2nd term in l1, forcing a placeholder classifier to be the 2nd closest to the known sample.
        loss3 = criterion(dummpyoutputs, dummytargets)
        # combining all losses
        loss = 0.1*loss1+args.lamda1*loss2+args.lamda2*loss3

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        newtargets = torch.cat((prehalflabels, laterhalflabels), 0)
        total += newtargets.size(0)
        correct += predicted.eq(newtargets).sum().item()

        # modification msuter: modify console info output
        print(f'\rBatch {idx+1}/{len(trainloader)} - Avg. Loss: {train_loss/(idx+1):.3f} - Accuracy: {100.0 * (correct/total):.3f} ({correct}/{total}) - L1: {loss1.item():.3f}, L2: {loss2.item():.3f}, L3: {loss3.item():.3f}', end="", flush=True)
    logger.debug('\n')
    # modification msuter: return training metrics for logging reasons
    return (train_loss/len(trainloader)), (correct/total)


# modification msuter: extend signature by argparse.Namespace object for accessing arguments and device
def computebias(dataloader, net, args, device):
    """Bias computation to tune the dummy (background class) logit magnitude"""
    bias = np.array([])

    # Calculates difference between max known logits and max dummy logit
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        dummylogit = dummypredict(net, inputs, args).view(-1)
        logits, _ = net(inputs)
        maxlogit, _ = torch.max(logits.clone(), dim=1)
        bias = np.append(bias, (maxlogit-dummylogit).detach().cpu().numpy())

    bias = np.sort(bias)  # torch.sort(bias)
    # Returns bias coresponding to the 5th percentile element, so that 95% of the data is classified as knowns.
    return bias[int(len(bias)*0.05)]


# modification msuter: delete unneeded getlogits() function
# modification msuter: delete unneeded getflowerplotdata()
# modification msuter: delete unneeded flowerplot() functions
# modification msuter: delete unneeded getdummyconf() function
# modification msuter: delete unneeded getprediction() function
# modification msuter: delete unneeded getAUC() function
# modification msuter: delete unneeded gettAccuracy() function


"""
Comments function "valdummy()":
In the model validation procedure, the same evaluation metric as proposed by the authors is used.
However, since the original implementation assumes the validation dataset coming in form of an
object of class "CIFAR10" (see file "cifar10_relabel.py" in author repo), the function "valdummy()"
needs to be modified. The modifications are introduced by a comment beginning with
"modification msuter: ...".
"""


def valdummy(val_logits, val_targets):
    # modification msuter: compute and return CONF_AUC as well as CONF_DeltaP instead of using flags
    auc_list = []
    auc_list_deltaP = []

    # modification msuter: prepare logits and targets outside of the function
    kkc_idx = val_targets >= 0
    kuc_idx = ~kkc_idx

    closelogits = val_logits[kkc_idx.squeeze()]
    openlogits = val_logits[kuc_idx.squeeze()]

    for temperature in [1024.0]:
        # modification msuter: operate directly on torch tensors instead of using a loop
        embeddings_close = nn.functional.softmax(
            closelogits/temperature, dim=1)
        embeddings_open = nn.functional.softmax(openlogits/temperature, dim=1)

        closeconf_auc = embeddings_close[:, -1].clone().detach().cpu().numpy()
        closelabel_auc = np.ones_like(closeconf_auc)

        openconf_auc = embeddings_open[:, -1].clone().detach().cpu().numpy()
        openlabel_auc = np.zeros_like(openconf_auc)

        totalbinary_auc = np.hstack([closelabel_auc, openlabel_auc])
        totalconf_auc = np.hstack([closeconf_auc, openconf_auc])
        auc1 = roc_auc_score(1-totalbinary_auc, totalconf_auc)
        auc2 = roc_auc_score(totalbinary_auc, totalconf_auc)
        auc_list.append(np.max([auc1, auc2]))

        logger.info(
            f'AUC of Confidence:\t Temperature: {temperature}\t AUC by Confidence: [{auc1}, {auc2}] \t AUC Added: {np.max([auc1, auc2])}')

        dummyconf_close_deltaP = embeddings_close[:, -
                                                  1].clone().detach().view(-1, 1)
        maxknownconf_close_deltaP, _ = torch.max(
            embeddings_close[:, :-1], dim=1)
        maxknownconf_close_deltaP = maxknownconf_close_deltaP.clone().detach().view(-1, 1)
        closeconf_deltaP = (dummyconf_close_deltaP -
                            maxknownconf_close_deltaP).cpu().numpy()
        closelabel_deltaP = np.ones_like(closeconf_deltaP)

        dummyconf_open_deltaP = embeddings_open[:, -
                                                1].clone().detach().view(-1, 1)
        maxknownconf_open_deltaP, _ = torch.max(embeddings_open[:, :-1], dim=1)
        maxknownconf_open_deltaP = maxknownconf_open_deltaP.clone().detach().view(-1, 1)
        openconf_deltaP = (dummyconf_open_deltaP -
                           maxknownconf_open_deltaP).cpu().numpy()
        openlabel_deltaP = np.zeros_like(openconf_deltaP)

        totalbinary_deltaP = np.hstack(
            [closelabel_deltaP.squeeze(), openlabel_deltaP.squeeze()])
        totalconf_deltaP = np.hstack(
            [closeconf_deltaP.squeeze(), openconf_deltaP.squeeze()])

        auc1_deltaP = roc_auc_score(1-totalbinary_deltaP, totalconf_deltaP)
        auc2_deltaP = roc_auc_score(totalbinary_deltaP, totalconf_deltaP)
        auc_list_deltaP.append(np.max([auc1_deltaP, auc2_deltaP]))

        logger.info(
            f'AUC of DeltaP:\t Temperature: {temperature}\t AUC by Confidence: [{auc1_deltaP}, {auc2_deltaP}] \t AUC Added: {np.max([auc1_deltaP, auc2_deltaP])}')

    return (np.max(np.array(auc_list)), np.max(np.array(auc_list_deltaP)))

# modification msuter: reimplementation of getConfidence() function


def getConfidence(logits, targets, unknown_target=-1):
    """
    Args:
        logits (torch.Tensor): logits for all samples of the dataset, with the last logit per sample referencing the dummy logit (C+1 classes)
        targets (torch.Tensor): targets for all samples of the datasest
        unknown_target (int): the target of samples associated with the unknown class

    Modifications:
    This reimplementation follows the basic idea of the original getConficence() function. Concretely, it
    computes the SoftMax confidences for known and unknown samples according to the VAST implementation 
    (see https://github.com/Vastlab/vast/blob/70ede97ae05b47c97536738277a11f2cb289afd1/vast/losses/metrics.py).
    It additionally computes a slightly altered dummy confidence as proposed in the original implementation
    (see https://gitlab.ifi.uzh.ch/aiml/projects/placeholder-open-set/-/blob/main/proser_unknown_detection.py)
    and performs a weighting of the confidences to counteract class imbalance.

    """
    assert unknown_target in targets, "not an open set dataset"

    with torch.no_grad():
        knowns = targets != unknown_target
        pred = torch.nn.functional.softmax(logits, dim=1)

        # sum softmax confidence of the correct class for known samples
        confidence_knowns = 0.0
        confidence_vast_unknowns = 0.0
        confidence_dummy_unknowns = 0.0

        confidence_knowns += torch.sum(pred[knowns, targets[knowns].long()])
        no_knowns = torch.sum(knowns)

        # according to VAST, confidence computation for unknown samples in case of a BG class is 1 - max(confidence[:C])
        confidence_vast_unknowns = torch.sum(
            1.0 - torch.max(pred[~knowns, :-1], dim=1)[0])
        confidence_dummy_unknowns = torch.sum(pred[~knowns, -1])
        no_unknowns = torch.sum(~knowns)

        logger.info(
            f'Weighted Avg. Confidence (VAST): {torch.mean(torch.tensor([(confidence_knowns/no_knowns).tolist(), (confidence_vast_unknowns/no_unknowns).tolist()]))} - Weighted Avg. Confidence (Dummy): {torch.mean(torch.tensor([(confidence_knowns/no_knowns).tolist(), (confidence_dummy_unknowns/no_unknowns).tolist()]))}')

    return confidence_knowns, confidence_vast_unknowns, confidence_dummy_unknowns, no_knowns, no_unknowns

# modification msuter: delete unneeded getF1score() function
# modification msuter: delete unneeded extract() function
# modification msuter: delete unneeded ccr_fpr_threshold() function
# modification msuter: delete unneeded ccr_fpr_diff() function
# modification msuter: delete unneeded testdummy() function


# modification msuter: extend signature by various parameters needed to invoke the training logic from the calling Proser class
def finetune_proser(net, epoch, args, train_loader, device):
    """
    modifications msuter:
     - delete code block responsible for loading and extending the pretrained network (already done in calling Proser class)
     - move seed configuration logic into the calling Proser class
     - move training loop into the calling Proser class to simplify evaluation process
     - delete unneeded variable 'bestF1' as well as function 'flowerplot()'
    """

    epoch_loss, epoch_accuracy = traindummy(
        epoch, net, args, train_loader, device)

    # Bias computation performed worse on ccr/fpr (currently done on training data)
    # modification msuter: add logging statements and option for performing bias computation
    if args.compute_bias == 'True':
        logger.debug('\n')
        logger.info(f'Starting Bias Computation')
        bias = computebias(train_loader, net, args=args, device=device)
    else:
        bias = 0.
    logger.info(f'Bias epoch {epoch+1}: \t {bias}')
    return bias, epoch_loss, epoch_accuracy


# modification msuter: delete unneeded main functionality (already done in calling Proser class)
