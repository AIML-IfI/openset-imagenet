import torch


"""
The source code is copied from the repo/branch:

  https://github.com/Vastlab/vast,
  Akshay Dhamija,
  Various Algorithms & Software Tools (VAST),
  version: 0.0.1
  branch: https://github.com/Vastlab/vast/blob/70ede97ae05b47c97536738277a11f2cb289afd1/vast/losses/metrics.py#L102

The code is associated with the following papers:

  Bendale, Abhijit and Boult, Terrance E,
  Towards open set deep networks,
  Proceedings of the IEEE conference on computer vision and pattern recognition,
  1563--1572, 2016

  Dhamija, Akshay R., Guenther, Manuel, and Boult, Terrance. E.,
  Reducing network agnostophobia,
  In Advances in Neural Information Processing Systems (NeurIPS),
  9157-9168, 2018

  Rudd, Ethan M and Jain, Lalit P and Scheirer, Walter J and Boult, Terrance E,
  The extreme value machine,
  IEEE transactions on pattern analysis and machine intelligence,
  40/3, 762-768, 2017

"""


def split_confidence(logits, target, negative_offset=0., unknown_class=-1):
    """Measures the softmax confidence of the correct class for known samples and for unknown samples:

    * with unknown_class = -1: 1 + negative_offset - max(confidence)
    * with unknown_class =  C: 1 - max(confidence[:C]) for unknown samples

    Parameters:

        logits: the output of the network, must be logits

        target: the vector of true classes; can be -1 for unknown samples

        negative_offset: the value to be added to the unknown confidence to turn the maximum to one, usually 1/C with C being the number of classes

        unknown_class: The class index that should be considered the unknown class; can be -1 or C

    Returns a tuple with four entries:

        known_confidence: the sum of the confidence values for the known samples

        unknown_confidence: the sum of the confidence values for the unknown samples

        known_samples: The total number of considered known samples in this batch

        unknown_samples: The total number of considered unknown samples in this batch
    """

    with torch.no_grad():
        known = target != unknown_class

        pred = torch.nn.functional.softmax(logits, dim=1)

        known_confidence = 0.
        unknown_confidence = 0.
        if torch.sum(known):
            known_confidence = torch.sum(pred[known, target[known]])
        if torch.sum(~known):
            if unknown_class == -1:
                unknown_confidence = torch.sum(
                    1. + negative_offset - torch.max(pred[~known], dim=1)[0])
            else:
                unknown_confidence = torch.sum(
                    1. - torch.max(pred[~known, :unknown_class], dim=1)[0])

    return known_confidence, unknown_confidence, torch.sum(known), torch.sum(~known)
