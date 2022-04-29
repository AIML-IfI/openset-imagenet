import torch

def fgsm_attack(x, epsilon, grad, device):
    """
    Generates and adversarial sample (using FGSM-method) and its corresponding label as known-unknown.
    Pixel values are clipped between [0,1].
    Parts taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    Args:
        x: Clean image.
        epsilon: Per-pixel attack magnitude.
        grad: Loss gradient with respect to sample x.
        device: Current cuda device.

    Returns: Tuple (adversarial sample, label).
    """
    # Get the element-wise sign of the gradient
    sign_data_grad = grad.sign()
    # Create the perturbed image
    perturbed_x = x + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_x = torch.clamp(perturbed_x, min=0.0, max=1.0)
    new_target = torch.ones(x.shape[0], device=device) * -1.0
    return perturbed_x, new_target


def decay_epsilon(start_eps, mu, curr_epoch, wait_epochs, lower_bound=0.01):
    """
    Decays using a base [mu], every [wait_epochs] a [start_eps] to a minimum value of [lower_bound].
    Pixel values are clipped between [0,1].
    Args:
        start_eps: Initial epsilon value.
        mu: Base of the decaying function.
        curr_epoch: Current training epoch.
        wait_epochs: Number of epochs to wait for every decay.
        lower_bound: Minimum epsilon to return.

    Returns:
        New epsilon value.
    """
    return max(start_eps * mu**(curr_epoch//wait_epochs), lower_bound)


def add_random_noise(x, epsilon, device):
    """
    Generates a sample perturbed with random noise and its corresponding label as known-unknown.
    Pixel values are clipped between [0,1].
    Args:
        x: Clean image.
        epsilon: Per-pixel noise magnitude.
        device: Current cuda device.

    Returns: Tuple (noisy image, label)
    """
    noise = torch.sign(torch.randn(x.shape, device=device))
    noisy_x = torch.clamp(x + epsilon*noise, min=0.0, max=1.0)
    new_target = torch.ones(x.shape[0], device=device) * -1.0
    return noisy_x, new_target


def add_gaussian_noise(x, loc, std, device):
    """
    Generates a sample perturbed with gaussian noise and its corresponding label as known-unknown.
    Pixel values are clipped between [0,1].
    Args:
        x: Clean image.
        loc: Mean of the gaussian distribution.
        std: Standard deviation of the gaussian distribution.
        device: Current cuda device.

    Returns: Tuple (noisy image, label)
    """
    noise = torch.empty(x.shape, device=device).normal_(mean=loc, std=std)
    noisy_x = torch.clamp(x + noise, min=0.0, max=1.0)
    new_target = torch.ones(x.shape[0], device=device) * -1.0
    return noisy_x, new_target
