from models.gatel0rd import HeavisideST
import torch

heaviside_gate = HeavisideST.apply


def batch_to_skip_inputs(xs, hs, add_inps=None):
    assert len(xs.shape) == len(hs.shape) == 2, "Only batch is accepted - no sequence data"
    if add_inps is not None:
        return torch.cat((torch.cat((xs, add_inps), dim=1), hs), dim=1)
    return torch.cat((xs, hs), dim=1)


def sequence_to_skip_data_gradfree(xs, gs, hs, add_inps=None, ignore_single_steps=True):
    with torch.no_grad():
        skip_inps, skip_tars = sequence_to_skip_data(xs, gs, hs, add_inps, ignore_single_steps)
    return skip_inps, skip_tars


def sequence_to_skip_data(xs, gs, hs, add_inps=None, ignore_single_steps=True):
    """
    Creates skip input training data from sequences of inputs
    :param xs: batch of lower level network input sequences (shape: seq_len x batch_size x input_dim)
    :param gs: batch of corresponding binary update gate openings (shape: seq_len x batch_size x latent_dim)
    :param hs: batch of corresponding latent states (shape: seq_len x batch_size x latent_dim)
    :param add_inps: optional additional information appended to inputs (shape: seq_len x batch_size x add_input_dim)
    :param ignore_single_steps: skips of length 1 are ignored
    :return: batch of skip network inputs (shape: new_batch_size x (input_dim + latent_dim + add_input_dim)),
             batch of skip network targets (shape: new_batch_size x input_dim)
    """

    S, B, D = xs.shape

    targets = torch.zeros((S, B, D + 1))

    if add_inps is not None:
        inputs = torch.cat((torch.cat((xs, add_inps), dim=2), hs), dim=2)
    else:
        inputs = torch.cat((xs, hs), dim=2)

    g_mask = None

    input_output_slice = torch.ones((S, B))

    for t in range(S):
        x_t = xs[t, :, :]
        g_t = gs[t, :, :]
        targets, g_mask = skip_data_gen_step(x_t, g_t, targets, t, g_mask)

        input_output_slice[t, :] = (1 - heaviside_gate(g_t.sum(dim=1))).detach()

    flat_slices = input_output_slice.flatten()
    skip_inps = inputs.flatten(0, 1)[flat_slices > 0, :]
    skip_tars = targets.flatten(0, 1)[flat_slices > 0, :]

    new_skip_tars = skip_tars[:, :(skip_tars.shape[1]-1)].clone()


    if not ignore_single_steps:
        next_targets = xs.clone()
        left_out_slices = input_output_slice[:(S-1), :].clone().flatten()
        next_inputs = next_targets[1:, :]
        left_out_inps = inputs[:(S-1), :].flatten(0, 1)[left_out_slices == 0, :]
        left_out_tars = next_inputs[:S, :].flatten(0, 1)[left_out_slices == 0, :]
        skip_inps = torch.cat((skip_inps, left_out_inps), 0)
        new_skip_tars = torch.cat((new_skip_tars, left_out_tars), 0)

    return skip_inps, new_skip_tars


def skip_data_gen_step(x_t, g_t, targets, t, g_mask=None):

    B, D = g_t.shape
    B2, I = x_t.shape
    S, B3, I2 = targets.shape

    assert B == B2 == B3, "Need same batch dimension for x_t, g_t, g_mask, and targets"
    assert I + 1 == I2, "Need same input dim for x_t and targets"
    assert S > t, "We must not be finished with the sequence"

    if g_mask is not None:
        S2, B4, D2 = g_mask.shape
        assert B == B4, "Need same batch dimension for g_t and g_mask"
        assert I2 == D2, "Need same latent dim for g_mask and targets"
        assert S2 == t, "g_mask needs to fit t"
    else:
        assert t == 0, "Only no g_mask for first time step"

    S_t = t + 1

    g_act = heaviside_gate(g_t.sum(dim=1, keepdim=True))  # Sum and clamp gate activations

    if S_t == S:  # End of sequence, all gates should be open
        g_act = heaviside_gate(g_act + torch.ones((B, 1))).detach()  # Detached because this is irrespective of gating
        x_t_hat = torch.cat((x_t, torch.ones(B, 1)), dim=1)  # Add ones to input to mark ending
    else:
        x_t_hat = torch.cat((x_t, torch.zeros(B, 1)), dim=1)  # Add zeros to input to mark not done

    g_seq = g_act.expand(S_t, B, I2)  # Reshape to right dimension

    if t == 0:
        g_mask_prime = torch.ones((1, B, I2))
    else:
        g_mask_prime = torch.cat((g_mask, torch.ones((1, B, I2))), dim=0)

    updates = g_seq * g_mask_prime

    if S_t == S:
        updates_padded = updates
    else:
        updates_padded = torch.cat((updates, torch.zeros((S - S_t, B, I2))), dim=0)

    x_seq = x_t_hat.expand(S, B, I2)

    new_targets = (1 - updates_padded) * targets + updates_padded * x_seq

    new_g_mask = (1 - updates) * g_mask_prime

    return new_targets, new_g_mask