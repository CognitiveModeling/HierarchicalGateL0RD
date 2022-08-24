import torch
import math

_LOG_2PI = math.log(2 * math.pi)

def gaussian_nll_weight_by_var(mean, var, target, beta):
    """ Beta-NLL loss from https://github.com/martius-lab/beta-nll"""

    ll = -0.5 * ((target - mean) ** 2 / var + torch.log(var) + _LOG_2PI)
    weight = var.detach() ** beta

    return -torch.sum(ll * weight, axis=-1)
