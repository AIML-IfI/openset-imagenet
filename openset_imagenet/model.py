""" ResNet50, parts taken from VAST: https://github.com/Vastlab/vast/tree/main/vast/architectures"""
from torchvision import models
from torch.nn.parallel import DistributedDataParallel
import torch
from torch import nn
import random
import numpy as np
import pathlib
import vast

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
    def __init__(self, dummy_count, fc_layer_dim, resnet_base, loss_type):
        super(ResNet50Proser, self).__init__()
        self.dummy_count = dummy_count
        # add a dummy classifier for unknown classes
        self.dummy_classifier = nn.Linear(fc_layer_dim, dummy_count)
        self.resnet_base = resnet_base
        self.loss_type=loss_type

    def first_blocks(self, x):
        """Calls the first three blocks of the model
        This repeats some functionality of the original ResNet implementation found here:
        https://github.com/pytorch/vision/blob/ad2eceabf0dcdb17a25d84da62492825a2c770a2/torchvision/models/resnet.py

        Note: for consistency reasons with the original source code of Zhou et al. (2021), the
        manifold mixup is performed after the third group of blocks (i.e. layer3). By following
        this approach, the manifold mixup is performed after the penultimate group/layer
        """

        x = self.resnet_base.resnet_base.conv1(x)
        x = self.resnet_base.resnet_base.bn1(x)
        x = self.resnet_base.resnet_base.relu(x)
        x = self.resnet_base.resnet_base.maxpool(x)

        x = self.resnet_base.resnet_base.layer1(x)
        x = self.resnet_base.resnet_base.layer2(x)
        x = self.resnet_base.resnet_base.layer3(x)


        return x

    def last_blocks(self, x):
        """Calls the last blocks of the model, and returns the deep features, the logits and the results of the dummy classifier
        This repeats some functionality of the original ResNet implementation found here:
        https://github.com/pytorch/vision/blob/ad2eceabf0dcdb17a25d84da62492825a2c770a2/torchvision/models/resnet.py

        Note: for consistency reasons with the original source code of Zhou et al. (2021), the
        manifold mixup is performed after the third group of blocks (i.e. layer3). By following
        this approach, the manifold mixup is performed after the penultimate group/layer
        """
        x = self.resnet_base.resnet_base.layer4(x)

        x = self.resnet_base.resnet_base.avgpool(x)

        x = torch.flatten(x, 1)

        features = self.resnet_base.resnet_base.fc(x)

        # apply our standard output layer
        logits = self.resnet_base.logits(features)

        if self.loss_type == "garbage":
            # for garbage class, we remove the logit for the unknown class -- since we will add another one below
            logits = logits[:,:-1]

        # apply our dummy layer, get only the maximum output
        dummy = torch.max(self.dummy_classifier(features), dim=1)[0]
        return logits, dummy, features

    def forward(self, image):
        """Extracts the logits, the dummy classiifers and the deep features for the given input """
        intermediate_features = self.first_blocks(image)
        return self.last_blocks(intermediate_features)

def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = Falsegg



def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    """ Saves a training checkpoint.

    Args:
        f_name(str): File name.
        model(torch module): Pytorch model.
        epoch(int): Current epoch.
        opt(torch optimizer): Current optimizer.
        best_score_(float): Current best score.
        scheduler(torch lr_scheduler): Pytorch scheduler.
    """
    # If model is DistributedDataParallel extracts the module.
    if isinstance(model, DistributedDataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    data = {"epoch": epoch + 1,
            "model_state_dict": state,
            "opt_state_dict": opt.state_dict(),
            "best_score": best_score_}
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, f_name)


def load_checkpoint(model, checkpoint, opt=None, scheduler=None):
    """ Loads a checkpoint, if the model was saved using DistributedDataParallel, removes the word
    'module' from the state_dictionary keys to load it in a single device. If fine-tuning model then
    optimizer should be none to start from clean optimizer parameters.

    Args:
        model (torch nn.module): Requires a model to load the state dictionary.
        checkpoint (Path): File path.
        opt (torch optimizer): An optimizer to load the state dictionary. Defaults to None.
        device (str): Device to load the checkpoint. Defaults to 'cpu'.
        scheduler (torch lr_scheduler): Learning rate scheduler. Defaults to None.
    """
    file_path = pathlib.Path(checkpoint)
    if file_path.is_file():  # First check if file exists
        checkpoint = torch.load(file_path, map_location=vast.tools._device)
        key = list(checkpoint["model_state_dict"].keys())[0]
        # If module was saved as DistributedDataParallel then removes the world "module"
        # from dictionary keys
        if key[:6] == "module":
            new_state_dict = OrderedDict()
            for k_i, v_i in checkpoint["model_state_dict"].items():
                key = k_i[7:]  # remove "module"
                new_state_dict[key] = v_i
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if opt is not None:  # Load optimizer state
            opt.load_state_dict(checkpoint["opt_state_dict"])

        if scheduler is not None:  # Load scheduler state
            scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        return start_epoch, best_score
    else:
        raise Exception(f"Checkpoint file '{checkpoint}' not found")
