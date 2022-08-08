# Taken from VAST: https://github.com/Vastlab/vast/tree/main/vast/architectures
from torchvision import models
import torch.nn as nn


class ResNet50(nn.Module):
    """Represents a ResNet50 model"""

    def __init__(self, fc_layer_dim=1000, out_features=1000, logit_bias=True):
        """ Builds a ResNet model, with deep features and logits layers.

        Args:
            fc_layer_dim: Deep features dimension.
            out_features: Logits dimension.
            logit_bias: True to use bias term in the logits layer.
        """
        super(ResNet50, self).__init__()
        resnet_base = models.resnet50(pretrained=False)
        # Change the dimension of out features
        fc_in_features = resnet_base.fc.in_features
        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.resnet_base = resnet_base
        self.logits = nn.Linear(in_features=fc_layer_dim, out_features=out_features, bias=logit_bias)

    def forward(self, x):
        """ Forward pass

        Args:
            x: Input samples

        Returns:
            Logits and deep features of the samples.
        """
        features = self.resnet_base(x)
        logits = self.logits(features)
        return logits, features
