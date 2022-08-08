"""Helper functions to create perturbations"""

import torch


def fgsm_attack(image, epsilon, grad, device):
    """ Generates an adversarial sample and its corresponding label as negative. Pixel values are
    clipped in [0,1]. Parts taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html.

    Args:
        image(tensor): Tensor of clean images.
        epsilon(float): Per-pixel attack magnitude.
        grad(tensor): Loss' gradient with respect to sample x.
        device(device): Current cuda device.

    Returns:
         perturbed: Tensor containing adversarial samples.
         new_target: Tensor containing adversarial samples labels
    """
    sign_data_grad = grad.sign()
    perturbed = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed = torch.clamp(perturbed, min=0.0, max=1.0)
    new_target = torch.ones(image.shape[0], device=device) * -1.0
    return perturbed, new_target


def decay_epsilon(start_eps, mu, curr_epoch, wait_epochs, lower_bound=0.01):
    """ Decays an initial epsilon [start_eps], waiting a number of epochs [wait_epochs], using a
    base factor [mu]. Pixel values are clipped in [0,1].

    Args:
        start_eps(float): Initial epsilon value.
        mu(float): Base of the decaying function.
        curr_epoch(int): Current training epoch.
        wait_epochs(int): Number of epochs to wait for every decay.
        lower_bound(float): Minimum epsilon to return.

    Returns:
        New epsilon value.
    """
    return max(start_eps * mu**(curr_epoch//wait_epochs), lower_bound)


def add_random_noise(image, epsilon, device):
    """ Generates a sample perturbed with random noise and its corresponding label as negative.
    Pixel values are clipped between [0,1].

    Args:
        image(tensor): Tensor of clean images.
        epsilon(float): Per-pixel noise magnitude.
        device(device): Current cuda device.

    Returns:
        noisy_im: Tensor of perturbed images with random noise.
        new_target: Tensor of labels of perturbed images.
    """
    # TODO: Does this use the main seed?
    noise = torch.sign(torch.randn(image.shape, device=device))
    noisy_im = torch.clamp(image + epsilon*noise, min=0.0, max=1.0)
    new_target = torch.ones(image.shape[0], device=device) * -1.0
    return noisy_im, new_target


def add_gaussian_noise(image, loc, std, device):
    """
    Generates a sample perturbed with gaussian noise and its corresponding label as negative.
    Pixel values are clipped between [0,1].

    Args:
        image(tensor): Tensor of clean images.
        loc(float): Mean of the gaussian distribution.
        std(float): Standard deviation of the gaussian distribution.
        device(device): Current cuda device.

    Returns:
        noisy_im: Tensor of perturbed images with gaussian noise.
        new_target: Tensor of labels of perturbed images.
    """
    noise = torch.empty(image.shape, device=device).normal_(mean=loc, std=std)
    noisy_im = torch.clamp(image + noise, min=0.0, max=1.0)
    new_target = torch.ones(image.shape[0], device=device) * -1.0
    return noisy_im, new_target
