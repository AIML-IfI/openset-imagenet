from torchvision import models
import torch.nn as nn

# Taken from VAST: https://github.com/Vastlab/vast/tree/main/vast/architectures

class ResNet50(nn.Module):
    """Represents a  ResNet50 model"""

    def __init__(self, fc_layer_dim=1000, out_features=1000, logit_bias=True):
        """
        Builds a ResNet50 model, with deep features and logits layers.
        Args:
            fc_layer_dim: Deep features dimension.
            out_features: Logits dimension.
            logit_bias: True to use bias term in the logits layer.
        """
        super(ResNet50, self).__init__()
        net = models.resnet50(pretrained=False)
        # Change the number of out features
        n_features = net.fc.in_features
        net.fc = nn.Linear(in_features=n_features, out_features=fc_layer_dim)
        self.net = net
        # Disable bias on logits layer.
        self.logits = nn.Linear(in_features=fc_layer_dim, out_features=out_features, bias=logit_bias)

    def forward(self, x, features=True):
        """
        Forward pass
        Args:
            x: Input samples
            features: True if return deep features of the samples.

        Returns: Logits and deep features of the samples.
        """

        d_features = self.net(x)
        logits = self.logits(d_features)
        if features:
            return logits, d_features
        else:
            return logits
