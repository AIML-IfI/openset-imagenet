""" ResNet50, parts taken from VAST: https://github.com/Vastlab/vast/tree/main/vast/architectures"""
from torchvision import models
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
        resnet_base = models.resnet50(pretrained=False)
        # Change the dimension of out features
        fc_in_features = resnet_base.fc.in_features
        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.resnet_base = resnet_base
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
