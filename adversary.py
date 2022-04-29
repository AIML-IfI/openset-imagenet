import torch


# Part taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def fgsm_attack(x, epsilon, grad, device):

    # Collect the element-wise sign of the data gradient
    sign_data_grad = grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_x = x + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_x = torch.clamp(perturbed_x, min=0.0, max=1.0)
    new_target = torch.ones(x.shape[0], device=device) * -1.0
    return perturbed_x, new_target


def decay_epsilon(start_eps, mu, curr_epoch, wait_epochs, lower_bound=0.01):
    return max(start_eps * mu**(curr_epoch//wait_epochs), lower_bound)


def add_gaussian_noise(x, loc, std, device):
    noise = torch.empty(x.shape, device=device).normal_(mean=loc, std=std)
    noisy_x = torch.clamp(x + noise, min=0.0, max=1.0)
    new_target = torch.ones(x.shape[0], device=device) * -1.0
    return noisy_x, new_target


def add_random_noise(x, epsilon, device):
    noise = torch.sign(torch.randn(x.shape, device=device))
    noisy_x = torch.clamp(x + epsilon*noise, min=0.0, max=1.0)
    new_target = torch.ones(x.shape[0], device=device) * -1.0
    return noisy_x, new_target
