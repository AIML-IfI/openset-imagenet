import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet18_Feature(models.ResNet):

    def __init__(self, feature_dim, num_classes, **kwargs):
        """
        Args:
            feature_dim (int): number of dimensions of the feature space
            num_classes (int): number of dimensions of the logit space
            **kwargs: see docs http://pytorch.org/vision/master/generated/torchvision.models.resnet50.html
        """

        # initialization of superclass according to source code, see method 'def resnet152' on http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html
        super(ResNet18_Feature, self).__init__(block=BasicBlock,
                                               layers=[2, 2, 2, 2], num_classes=feature_dim, **kwargs)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x_features = super(ResNet18_Feature, self)._forward_impl(x)
        x = self.fc2(x_features)

        return x, x_features


class ResNet34_Feature(models.ResNet):

    def __init__(self, feature_dim, num_classes, **kwargs):
        """
        Args:
            feature_dim (int): number of dimensions of the feature space
            num_classes (int): number of dimensions of the logit space
            **kwargs: see docs http://pytorch.org/vision/master/generated/torchvision.models.resnet50.html
        """

        # initialization of superclass according to source code, see method 'def resnet152' on http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html
        super(ResNet34_Feature, self).__init__(block=BasicBlock,
                                               layers=[3, 4, 6, 3], num_classes=feature_dim, **kwargs)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x_features = super(ResNet34_Feature, self)._forward_impl(x)
        x = self.fc2(x_features)

        return x, x_features


class ResNet50_Feature(models.ResNet):

    def __init__(self, feature_dim, num_classes, **kwargs):
        """
        Args:
            feature_dim (int): number of dimensions of the feature space
            num_classes (int): number of dimensions of the logit space
            **kwargs: see docs http://pytorch.org/vision/master/generated/torchvision.models.resnet50.html
        """

        # initialization of superclass according to source code, see method 'def resnet50' on http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html#resnet50
        super(ResNet50_Feature, self).__init__(block=Bottleneck,
                                               layers=[3, 4, 6, 3], num_classes=feature_dim, **kwargs)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x_features = super(ResNet50_Feature, self)._forward_impl(x)
        x = self.fc2(x_features)

        return x, x_features


class ResNet101_Feature(models.ResNet):

    def __init__(self, feature_dim, num_classes, **kwargs):
        """
        Args:
            feature_dim (int): number of dimensions of the feature space
            num_classes (int): number of dimensions of the logit space
            **kwargs: see docs http://pytorch.org/vision/master/generated/torchvision.models.resnet50.html
        """

        # initialization of superclass according to source code, see method 'def resnet101' on http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html
        super(ResNet101_Feature, self).__init__(block=Bottleneck,
                                                layers=[3, 4, 23, 3], num_classes=feature_dim, **kwargs)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x_features = super(ResNet101_Feature, self)._forward_impl(x)
        x = self.fc2(x_features)

        return x, x_features


class ResNet152_Feature(models.ResNet):

    def __init__(self, feature_dim, num_classes, **kwargs):
        """
        Args:
            feature_dim (int): number of dimensions of the feature space
            num_classes (int): number of dimensions of the logit space
            **kwargs: see docs http://pytorch.org/vision/master/generated/torchvision.models.resnet50.html
        """

        # initialization of superclass according to source code, see method 'def resnet152' on http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html
        super(ResNet152_Feature, self).__init__(block=Bottleneck,
                                                layers=[3, 8, 36, 3], num_classes=feature_dim, **kwargs)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x_features = super(ResNet152_Feature, self)._forward_impl(x)
        x = self.fc2(x_features)

        return x, x_features
