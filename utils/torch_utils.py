import torch

def mean_euclidean_distance(a, b):
    return torch.mean(torch.sqrt(torch.sum(torch.pow(b - a, 2), 1)))

def compute_multivariate_normal_entropy(mus, sigmas):
    assert len(mus.shape) == 2 and len(sigmas.shape), "Expect batches of means and sigmas"
    normal = torch.distributions.Normal(mus, sigmas)
    diagn = torch.distributions.Independent(normal, 1)  # Wrap it with independent to make it multivariate
    return diagn.entropy()