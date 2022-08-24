import torch
import torch.nn as nn
from models.beta_nll_loss import gaussian_nll_weight_by_var
import models.rnn_helpers as rnn_helpers


class SkipNet(torch.nn.Module):
    """
    SkipNetwork is a deep MLP.
    Can be trained to predict normal distributions (gauss_net=True) using the beta NLL loss
    or simply predict vectors (gauss_net=False)
    """

    def __init__(self, input_dim, output_dim, feature_dim, num_layers,
                 gauss_net=False, nll_beta=0.0):
        
        super(SkipNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # Network has fan-in shape mapping on layer with feature dim
        self.input_to_feature = rnn_helpers.create_f_pre(
            f_pre_layers=(num_layers-1),
            input_dim=input_dim,
            feature_dim=feature_dim
        )

        # Linear read-out layer
        self.feature_to_output = rnn_helpers.create_f_post(
            f_post_layers=1,
            feature_dim=feature_dim,
            output_dim=output_dim
        )

        # Gaussian network also needs to predict the variance
        self.gauss_net = gauss_net
        if gauss_net:
            self.feature_to_sigma = rnn_helpers.create_f_post(
                f_post_layers=1,
                feature_dim=feature_dim,
                output_dim=output_dim
            )

        # Fixed hps for beta-NLL
        self.epsilon = 0.000001
        self.nll_beta = nll_beta
        self.ELU = nn.ELU()

        # loss
        self.MSE = nn.MSELoss()

    def forward(self, inp_batch):
        assert len(inp_batch.shape) == 2, "Need 2d input"
        feature_batch = self.input_to_feature(inp_batch)
        outputs = self.feature_to_output(feature_batch)
        sigma_batch = None
        if self.gauss_net:
            sigma_batch = self.ELU(self.feature_to_sigma(feature_batch)) + 1 + self.epsilon
        return outputs, sigma_batch

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-04)

    def loss(self, out, tar, sigmas):
        if self.gauss_net:
            return self.loss_gauss(out, tar, sigmas)
        return self.MSE(out, tar)

    def loss_gauss(self, out, tar, sigmas):
        return gaussian_nll_weight_by_var(mean=out, var=sigmas, target=tar, beta=self.nll_beta).mean()

    def NLL(self, mus, sigmas, tars, ignore_beta=True):
        beta = self.nll_beta
        if ignore_beta:
            beta = 0.0
        return gaussian_nll_weight_by_var(
            mean=mus.flatten(start_dim=0, end_dim=1),
            var=sigmas.flatten(start_dim=0, end_dim=1),
            target=tars.flatten(start_dim=0, end_dim=1),
            beta=beta
        ).mean()

