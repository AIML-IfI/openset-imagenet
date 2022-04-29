from torchvision import models
import torch.nn as nn

# Taken from VAST
# Note on softmax:
# If you would like to use nn.CrossEntropyLoss, you should pass the raw logits to the loss function
# (i.e. no final non-linearity), since nn.LogSoftmax and nn.NLLLoss will be called internally.
# However, if you would like to use nn.NLLLoss, you should add the nn.LogSoftmax manually.


class ResNet50(nn.Module):
    """Represents a  ResNet50 network todo: should include a parameter for pretrained? load from my saved net"""

    def __init__(self, fc_layer_dim=1000, out_features=1000, logit_bias=True):
        """
        
        Args:
            fc_layer_dim:
            out_features:
            logit_bias:
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

        Args:
            x:
            features:

        Returns:

        """
        # Second last fully connected layer (deep features)
        d_features = self.net(x)
        # last fully connected layer (logits)
        logits = self.logits(d_features)
        if features:
            # Return logits and features
            return logits, d_features
        else:
            # Return logits
            return logits
