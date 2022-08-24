import torch

'''
Helpers to create 
- f_pre, preprocessing MLP of inputs
- f_post, read out layers of network output
- f_init, "warm up" layers setting the initial latent state
- multiplicative sigmoidal gate, feed-forward net with two branches (linear & sigmoidal) whose outputs are multiplied
'''


def create_f_pre(f_pre_layers, input_dim, feature_dim):
    module_pre = torch.nn.ModuleList([])
    if f_pre_layers < 1:
        module_pre.append(torch.nn.Identity())
    h_dim = input_dim
    for pre_l in range(f_pre_layers):
        # Fan in type of network, decreasing features per layer
        pre_l_factor = pow(2, (f_pre_layers - pre_l - 1))
        module_pre.append(torch.nn.Linear(h_dim, pre_l_factor * feature_dim))
        module_pre.append(torch.nn.Tanh())
        h_dim = pre_l_factor * feature_dim
    return torch.nn.Sequential(*module_pre)


def create_f_init(f_init_layers, f_init_inputs, input_dim, feature_dim, latent_dim, last_layer_linear=False):
    input_dim_warm_up = input_dim * f_init_inputs
    feature_dim_warm_up = feature_dim
    warm_up_net = torch.nn.ModuleList([])
    for w in range(f_init_layers):
        w_factor = pow(2, (f_init_layers - w - 1))
        if w == (f_init_layers - 1):
            feature_dim_warm_up = latent_dim
        warm_up_net.append(torch.nn.Linear(input_dim_warm_up, w_factor * feature_dim_warm_up))
        if not last_layer_linear or w != (f_init_layers - 1):
            warm_up_net.append(torch.nn.Tanh())
        input_dim_warm_up = w_factor * feature_dim
    return torch.nn.Sequential(*warm_up_net)


def create_f_post(f_post_layers, feature_dim, output_dim):
    post_module = torch.nn.ModuleList([])
    in_post = feature_dim
    h_dim = feature_dim
    for post_l in range(f_post_layers):
        h_factor = pow(2, (f_post_layers - post_l - 2))
        if post_l == f_post_layers - 1:
            h_factor = 1
            h_dim = output_dim
        post_module.append(torch.nn.Linear(in_post, h_factor * h_dim))
        if post_l < (f_post_layers - 1):
            post_module.append(torch.nn.Tanh())
        in_post = h_factor * h_dim
    return torch.nn.Sequential(*post_module)


def create_multi_gate(input_dim, output_dim):
    gate_net = torch.nn.ModuleList([])
    gate_net.append(torch.nn.Linear(input_dim, output_dim))
    gate_net.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*gate_net)