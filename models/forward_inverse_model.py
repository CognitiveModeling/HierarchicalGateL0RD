# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import models.rnn_helpers as rnn_helpers
from models.gatel0rd import GateL0RDCell
from models.my_grucell import MyGRUCell
from models.beta_nll_loss import gaussian_nll_weight_by_var


class ForwardInverseModel(nn.Module):

    def __init__(
            self,
            obs_dim,
            action_dim,
            latent_dim,
            feature_dim,
            reg_lambda,
            num_layers_internal,
            f_FM_layers, # layers for forward model head
            f_IM_layers, # layers for inverse model head
            f_init_layers, # layers for latent state initialization
            f_init_inputs=1, # number of inputs used for latent state initializtaion
            num_layers=1, # number of stacked RNN layers
            gauss_network=False, # predict Gaussian distributions?
            additional_input_dim=0, # number of additional inputs concatenated to observation and action
            nll_beta=0.0, # HP beta NLL loss
            rnn_type='GateL0RD', # either 'GateL0RD' or 'GRU'
            FM_IM_loss_weight=0.5, # balancing forward and inverse prediction
    ):

        super(ForwardInverseModel, self).__init__()

        # dimensionality of inputs and outputs
        self.input_dim = obs_dim + action_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.add_input_dim = additional_input_dim

        # network structure
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.gauss_net = gauss_network

        # f_init initializes the latent state (warm up)
        self.use_warm_up = f_init_layers > 0 and f_init_inputs > 0
        self.warm_up_inputs = f_init_inputs
        self.warm_up_network_list = nn.ModuleList([])
        if self.use_warm_up:
            for _ in range(self.num_layers):
                self.warm_up_network_list.append(rnn_helpers.create_f_init(
                    f_init_layers=f_init_layers,
                    f_init_inputs=f_init_inputs,
                    input_dim=self.input_dim,
                    feature_dim=feature_dim,
                    latent_dim=latent_dim)
                )

        # The memory module: (stacked) RNN cells:
        cell_input = self.input_dim + self.add_input_dim
        self.cells = nn.ModuleList([])
        for _ in range(num_layers):
            if rnn_type == 'GateL0RD':
                self.cells.append(GateL0RDCell(
                    input_size=cell_input,
                    output_size=feature_dim,
                    hidden_size=latent_dim,
                    reg_lambda=reg_lambda,
                    num_layers_internal=num_layers_internal,
                    gate_noise_level=0.1
                ))
            elif rnn_type == 'GRU':
                self.cells.append(
                    MyGRUCell(
                        input_size=cell_input,
                        output_size=feature_dim,
                        hidden_size=latent_dim
                    )
                )
            cell_input = feature_dim

        # The forward model (FM) branch
        self.fc_FM = rnn_helpers.create_f_post(f_post_layers=f_FM_layers, feature_dim=feature_dim, output_dim=obs_dim)
        if self.gauss_net:
            self.fc_FM_sigma = rnn_helpers.create_f_post(f_post_layers=f_FM_layers,
                                                          feature_dim=feature_dim,
                                                          output_dim=obs_dim)

        # The inverse model (IM) branch
        overall_latent_dim = int(latent_dim * num_layers)

        # Sigmoidal gating plus read-out layers:
        self.fc_IM_gate1 = rnn_helpers.create_multi_gate(obs_dim + overall_latent_dim, obs_dim + overall_latent_dim)
        self.fc_IM_gate2 = rnn_helpers.create_multi_gate(obs_dim + overall_latent_dim, obs_dim + overall_latent_dim)

        self.fc_IM = rnn_helpers.create_f_init(f_init_layers=f_IM_layers, f_init_inputs=1,
                                               input_dim=obs_dim + overall_latent_dim, feature_dim=feature_dim,
                                               latent_dim=action_dim, last_layer_linear=True)
        if self.gauss_net:
            self.fc_IM_sigma = rnn_helpers.create_f_init(f_init_layers=f_IM_layers, f_init_inputs=1,
                                                         input_dim=obs_dim + overall_latent_dim,
                                                         feature_dim=feature_dim, latent_dim=action_dim,
                                                         last_layer_linear=True)

        # MSE Loss (vector predictions)
        self.MSE = nn.MSELoss()

        # fixed definitions for Gaussian networks
        self.ELU = nn.ELU()
        self.nll_beta = nll_beta
        self.epsilon = 0.000001

        # balance between action and obs prediction terms
        self.FM_IM_loss_weight = FM_IM_loss_weight

    def forward_n_step(
            self,
            obs_batch,
            action_batch,
            train_schedule,
            predict_o_deltas,
            factor_o_delta=-1,
            mode='imagination',
            predict_a_deltas=False,
            factor_a_delta=None,
            additional_info=None
    ):
        """
        Forward prediction in autoregressive mode.
        There are two modes for using predictions as inputs observation vs. imagination

                                    ┌──────o_t+1           a_t+1─────┐
                                    │       ▲                ▲       │
                                    │       │                │       │
                                    │    ┌──┴───┐        ┌───┴──┐    │
                                    │    │      │        │      │    │
                                    │    │ FM   │        │  IM  │    │
                                    │    │      │        │      │    │
                                    │    └──────┘        └──────┘    │
                                    │       ▲                ▲       │
                                    │       └───────┬────────┘       │
                                    │               │                │
                                    │     ┌────  h_t│                │
                                    │     │    ┌────┴─────┐          │
                                    │     │    │          │          │
                                    │     │    │          │          │
                                    │     │    │ GateL0RD │          │
                                    │     │    │          │          │
                                    │     └──► │          │          │
                                    │          └──────────┘          │
                                    │               ▲  ▲             │
                                    │      if       │  │    if       │
                                    └────►imagi-────┤  ├──obser-◄────┘
                                          nation    │  │  vation
                                                    │  │
                                                  o_t  a_t

        The predictions are only stochastically used according to train_schedule (ala scheduled sampling)

        :param obs_batch: batch of observations (shape: seq_len x batch_size x obs_dim)
        :param action_batch: batch of actions (shape: seq_len x batch_size x act_dim)
        :param train_schedule: scheduled sampling, determines probability for using real input
                               (np array, shape: seq_len x batch_size)
        :param predict_o_deltas: residual connections for obs? states whether network predicts o_t+1 or \Delta o_t+1
        :param factor_o_delta: when predicting deltas one can specifiy a constant k for
                            which the network predicts k * \Delta x_t+1
        :param mode: mode of forward pass:
                - imagination: actions are given, observations are predicted according to train_schedule
                - observation: observations are given, actions are predicted according to train_schedule
        :param predict_a_deltas: residual connections for actions? states whether network predicts a_t+1 or \Delta a_t+1
        :return: - batch of network outputs(shape: sequence length  X batch size X output dim)
                 - latent states (shape: layer number x sequence length  X batch size X latent dim)
                 - regularized gate activations \Theta (shape: layer number x sequence length  X batch size X latent dim)
                 - layer outputs (shape: layer number x sequence length  X batch size X feature dim)
                 - network outputs as deltas (shape: sequence length  X batch size X output dim)
                 - (potential) second output branch
                 - (potential) predicted variance of first output
                 - (potential) predicted variance of second output
        """

        seq_len, batch_size, obs_dim = obs_batch.size()
        _, _, act_dim = action_batch.size()


        # sample based on scheduled sampling probabilities
        sampling_rand = np.random.rand(seq_len, batch_size)
        sampling_schedule_np = np.clip(np.ceil(train_schedule - sampling_rand), 0, 1)

        if mode == 'observation':
            sampling_schedule = torch.unsqueeze(torch.from_numpy(sampling_schedule_np), dim=2).expand(
                seq_len,
                batch_size,
                act_dim
            ).float()
        else:
            assert mode == 'imagination', "Mode " + mode + " unknown!"
            sampling_schedule = torch.unsqueeze(torch.from_numpy(sampling_schedule_np), dim=2).expand(
                seq_len,
                batch_size,
                obs_dim
            ).float()

        # additional input besides observation and action?
        use_additional_info = additional_info is not None

        # keep track of all latent states (hs), layer outputs (ys) regularized gate activations
        # (gs_reg, i.e. Theta(s) in paper), network outputs for both output heads
        # (outs, delta_outs for FM output and outs2 for IM output):
        last_out = None
        last_act_out = None
        hs = []
        ys = []
        gs_reg = []
        outs = []
        outs_sigmas = []
        delta_outs = []
        outs2 = []
        outs2_sigmas = []

        if factor_a_delta is None:
            factor_a_delta = factor_o_delta # per default use same factor

        for t in range(seq_len):

            hs_at_t = []
            ys_at_t = []
            gs_reg_at_t = []

            # determine network input based on scheduled sampling
            if last_out is None or mode == 'observation':
                x_in_t = obs_batch[t, :, :]
            else:
                x_in_t = sampling_schedule[t, :, :] * obs_batch[t, :, :] + (1 - sampling_schedule[t, :, :]) * last_out

            # for planning the actions need to be concatenated
            if mode == 'imagination' or (mode == 'observation' and last_act_out is None):
                a_in_t = action_batch[t, :, :]
                x_in_t_full = torch.cat((a_in_t, x_in_t), 1)
            else:
                assert mode == 'observation'
                a_in_t = sampling_schedule[t, :, :] * action_batch[t, :, :] \
                         + (1 - sampling_schedule[t, :, :]) * last_act_out
                x_in_t_full = torch.cat((a_in_t, x_in_t), 1)

            if use_additional_info:
                x_in_t_full = torch.cat((x_in_t_full, additional_info[t, :, :]), 1)

            # input per layer
            x_lt = x_in_t_full

            # propagate the input through all layers
            for layer in range(self.num_layers):

                # get the last latent state
                if t > 0:
                    # there exists a previous latent state
                    last_h_l = hs[t - 1][layer, :, :]
                else:
                    # initialize the latent state, either through warm up or with zeroes
                    if self.use_warm_up:
                        last_h_l = self._warm_up(obs_batch=obs_batch, layer=layer, action_batch=action_batch)
                    else:
                        last_h_l = torch.zeros((batch_size, self.latent_dim))

                y_lt, h_lt, g_lt_reg = self._forward_per_layer(x_lt=x_lt, h_ltminus1=last_h_l, l=layer)

                # input to the next layer is the output of the current layer
                x_lt = y_lt

                # store all latent states and gate activations
                hs_at_t.append(h_lt)
                ys_at_t.append(y_lt)
                gs_reg_at_t.append(g_lt_reg)

            # stack the latent states and gate activations of all layers
            all_hs_at_t = torch.stack(hs_at_t)
            hs.append(all_hs_at_t)
            ys.append(torch.stack(ys_at_t))
            gs_reg.append(torch.stack(gs_reg_at_t))

            # FM: use the read-out layer to compute the outputs
            out_pre = self.fc_FM(x_lt)

            if self.gauss_net: # predicted variances for gauss nets
                outs_sigmas.append(self.ELU(self.fc_FM_sigma(x_lt)) + 1 + self.epsilon)

            # save the network outputs (and deltas)
            delta_outs.append(out_pre)
            last_out = out_pre
            if predict_o_deltas:
                last_out = x_in_t + factor_o_delta * out_pre
            outs.append(last_out)

            # IM: second output head for action prediction
            obs_inp_out2 = last_out

            if mode == 'observation' and (t + 1) < seq_len: # during observation all obs are visible
                obs_inp_out2 = obs_batch[t + 1, :, :]

            out2_input_pre = torch.cat((obs_inp_out2, all_hs_at_t.permute(1, 2, 0).flatten(1, 2)), 1)

            # multiplicative gating:
            out2_input = self.fc_IM_gate1(out2_input_pre) * self.fc_IM_gate2(out2_input_pre)
            last_out2 = self.fc_IM(out2_input)

            if self.gauss_net:
                outs2_sigmas.append(self.ELU(self.fc_IM_sigma(out2_input)) + 1 + self.epsilon)

            if predict_a_deltas:
                last_out2 = a_in_t + factor_a_delta * last_out2

            outs2.append(last_out2)
            last_act_out = last_out2

        # stack the latent states, gate activations, etc... over all time steps
        # me move the layer-dimension to the front, since its 1 in most cases
        hs_final = torch.stack(hs).permute([1, 0, 2, 3])
        ys_final = torch.stack(ys).permute([1, 0, 2, 3])
        gs_reg_final = torch.stack(gs_reg).permute([1, 0, 2, 3])

        sigmas_final = None
        sigmas2_final = None

        if self.gauss_net:
            sigmas_final = torch.stack(outs_sigmas)
            sigmas2_final = torch.stack(outs2_sigmas)

        return torch.stack(outs), hs_final, gs_reg_final, ys_final, \
               torch.stack(delta_outs), torch.stack(outs2), sigmas_final, sigmas2_final


    def predict_one_step(
            self,
            obs_batch,
            action_batch,
            predict_o_deltas,
            h_init,
            factor_o_delta=-1,
            predict_a_deltas=False,
            factor_a_delta=None,
            additional_info=None
    ):

        assert len(obs_batch.size()) == len(action_batch.size()) == 2, "Need batch for 1-step prediction, not sequences"

        # additional input besides observation and action?
        use_additional_info = additional_info is not None

        hs_at_t = []
        ys_at_t = []
        gs_reg_at_t = []

        x_in_t_full = torch.cat((action_batch, obs_batch), 1)

        if use_additional_info:
            x_in_t_full = torch.cat((x_in_t_full, additional_info[:, :]), 1)

        x_lt = x_in_t_full

        for layer in range(self.num_layers):

            if h_init is not None:
                last_h_l = h_init[layer, :, :]
            else:
                if self.use_warm_up:
                    last_h_l = self._warm_up(obs_batch=obs_batch.unsqueeze(0), layer=layer, action_batch=action_batch.unsqueeze(0))
                else:
                    last_h_l = torch.zeros((obs_batch.shape[0], self.latent_dim))

            y_lt, h_lt, g_lt_reg = self._forward_per_layer(x_lt=x_lt, h_ltminus1=last_h_l, l=layer)
            x_lt = y_lt
            hs_at_t.append(h_lt)
            ys_at_t.append(y_lt)
            gs_reg_at_t.append(g_lt_reg)

        hs = torch.stack(hs_at_t)
        ys = torch.stack(ys_at_t)
        gs_reg = torch.stack(gs_reg_at_t)
        out_pre = self.fc_FM(x_lt)

        out_sigmas = None
        if self.gauss_net:
            out_sigmas = self.ELU(self.fc_FM_sigma(x_lt)) + 1 + self.epsilon
        delta_out = out_pre
        last_out = out_pre
        if predict_o_deltas:
            last_out = obs_batch + factor_o_delta * out_pre

        obs_inp_out2 = obs_batch
        out2_input_pre = torch.cat((obs_inp_out2, hs.permute(1, 2, 0).flatten(1, 2)), 1)
        out2_input = self.fc_IM_gate1(out2_input_pre) * self.fc_IM_gate2(out2_input_pre)
        last_out2 = self.fc_IM(out2_input)

        out_sigmas2 = None
        if self.gauss_net:
            out_sigmas2 = self.ELU(self.fc_IM_sigma(out2_input)) + 1 + self.epsilon

        if predict_a_deltas:
            last_out2 = action_batch + factor_a_delta * last_out2
        return last_out, hs, gs_reg, ys, delta_out, last_out2, out_sigmas, out_sigmas2

    def _warm_up(self, obs_batch, action_batch, layer):
        """
        Implements f_init call to initialize latent state
        :param obs_batch: batch of observations (shape: seq_len x batch_size x obs_dim)
        :param action_batch: batch of actions (shape: sequence length  X batch size X action dim)
        :param layer: index l of layer when using a stacked GateL0RD
        :return: h_0 for layer l
        """
        seq_len, _, _ = obs_batch.size()
        warm_up_input = torch.cat((action_batch[0, :, :], obs_batch[0, :, :]), 1)
        for t_warm_up in range(self.warm_up_inputs - 1):
            next_warm_up_input = obs_batch[t_warm_up + 1, :, :]
            warm_up_input_t_plus1 = torch.cat((action_batch[t_warm_up + 1, :, :], next_warm_up_input), 1)
            warm_up_input = torch.cat((warm_up_input, warm_up_input_t_plus1), 1)
        return self.warm_up_network_list[layer](warm_up_input)


    def warm_up_all_layers(self, obs_batch, action_batch):
        """
        Implements f_init call for all layers when using stacked GateL0RD cells
        :param obs_batch: batch of observations (shape: seq_len x batch_size x obs_dim)
        :param action_batch: batch of actions (shape: sequence length  X batch size X action dim)
        :return: h_0 for layer l
        """

        h_list = []
        # Propagate the input through all layers
        for layer in range(self.num_layers):

            if self.use_warm_up:
                last_h_l = self._warm_up(
                    obs_batch=obs_batch.unsqueeze(0),
                    layer=layer,
                    action_batch=action_batch.unsqueeze(0)
                )
            else:
                last_h_l = torch.zeros((obs_batch.shape[0], self.latent_dim))
            h_list.append(last_h_l)
        return torch.stack(h_list)

    def _forward_per_layer(self, x_lt, h_ltminus1, l):
        """
        Forward pass one step through one layer of GateL0RD, i.e. pass through g-, r-, p- and o-network of one cell
        """
        return self.cells[l](x_lt, h_ltminus1)


    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-04)


    def loss(self, out, tar, g_regs, out2, tar2, sigmas=None, sigmas2=None):
        if not self.gauss_net:
            return self.loss_MSE(out, tar, g_regs, out2, tar2)
        else:
            return self.loss_gauss(out, tar, g_regs, sigmas, out2, tar2, sigmas2)


    def loss_MSE(self, out, tar, g_regs, out2, tar2):
        MSE_loss = (1.0 - self.FM_IM_loss_weight) * self.MSE(out, tar) \
                   + self.FM_IM_loss_weight * self.MSE(out2, tar2)
        return self.cells[0].loss(MSE_loss, g_regs)


    def loss_gauss(self, out, tar, g_regs, sigmas, out2, tar2, sigmas2):
        mean_nll = (1.0 - self.FM_IM_loss_weight) * self.NLL(out, sigmas, tar) \
                   + self.FM_IM_loss_weight * self.NLL(out2, sigmas2, tar2)
        return self.cells[0].loss(mean_nll, g_regs)


    def NLL(self, mus, sigmas, tars, ignore_beta=False):
        beta = self.nll_beta
        if ignore_beta:
            beta = 0.0
        return gaussian_nll_weight_by_var(
            mean=mus.flatten(start_dim=0, end_dim=1),
            var=sigmas.flatten(start_dim=0, end_dim=1),
            target=tars.flatten(start_dim=0, end_dim=1),
            beta=beta
        ).mean()
