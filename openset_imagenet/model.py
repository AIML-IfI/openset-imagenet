""" ResNet50, parts taken from VAST: https://github.com/Vastlab/vast/tree/main/vast/architectures"""
from torchvision import models
import torch
from torch import nn

class ResNet50(nn.Module):
    """Represents a ResNet50 model"""

    def __init__(self, fc_layer_dim=1000, out_features=1000, logit_bias=True):
        """ Builds a ResNet model, with deep features and logits layers.

        Args:
            fc_layer_dim(int): Deep features dimension.
            out_features(int): Logits dimension.
            logit_bias(bool): True to use bias term in the logits layer.
        """
        super(ResNet50, self).__init__()

        self.number_of_classes = out_features

        # Change the dimension of out features
        self.resnet_base = models.resnet50(pretrained=False)
        fc_in_features = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.logits = nn.Linear(
            in_features=fc_layer_dim,
            out_features=out_features,
            bias=logit_bias)

    def forward(self, image):
        """ Forward pass

        Args:
            image(tensor): Tensor with input samples

        Returns:
            Logits and deep features of the samples.
        """
        features = self.resnet_base(image)
        logits = self.logits(features)
        return logits, features


class ResNet50Proser(nn.Module):
    """Implements functionality for the PROSER approach into ResNet50"""
    def __init__(self, dummy_count, fc_layer_dim=1000, resnet_base = None):
        super(ResNet50Proser, self).__init__()
        self.dummy_count = dummy_count
        # add a dummy classifier for unknown classes
        self.dummy_classifier = nn.Linear(fc_layer_dim, dummy_count)
        self.resnet_base = resnet_base

    def first_blocks(self, x):
        """Calls the first three blocks of the model
        This repeats some functionality of the original ResNet implementation found here:
        https://github.com/pytorch/vision/blob/ad2eceabf0dcdb17a25d84da62492825a2c770a2/torchvision/models/resnet.py

        Note: for consistency reasons with the original source code of Zhou et al. (2021), the
        manifold mixup is performed after the third group of blocks (i.e. layer3). By following
        this approach, the manifold mixup is performed after the penultimate group/layer
        """
        x = self.resnet_base.conv1(x)
        x = self.resnet_base.bn1(x)
        x = self.resnet_base.relu(x)
        x = self.resnet_base.maxpool(x)

        x = self.resnet_base.layer1(x)
        x = self.resnet_base.layer2(x)
        x = self.resnet_base.layer3(x)
        return x

    def last_blocks(self, x):
        """Calls the last blocks of the model, and returns the deep features, the logits and the results of the dummy classifier
        This repeats some functionality of the original ResNet implementation found here:
        https://github.com/pytorch/vision/blob/ad2eceabf0dcdb17a25d84da62492825a2c770a2/torchvision/models/resnet.py

        Note: for consistency reasons with the original source code of Zhou et al. (2021), the
        manifold mixup is performed after the third group of blocks (i.e. layer3). By following
        this approach, the manifold mixup is performed after the penultimate group/layer
        """
        x = self.resnet_base.layer4(x)

        x = self.resnet_base.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.resnet_base.fc(x)

        # apply our standard output layer
        logits = self.logits(features)
        # apply our dummy layer, get only the maximum output
        dummy = torch.max(self.dummy(features), dim=1)
        return logits, dummy, features

    def forawrd(self, image):
        """Extracts the logits, the dummy classiifers and the deep features for the given input """
        intermediate_features = self.first_blocks(image)
        return self.last_blocks(intermediate_features)
