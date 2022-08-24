import torch
import torch.nn.functional as F
import random
import numpy as np

def add_gaze_based_noise(obs_t, obs_tplus1, gaze_noise_sd, focus_noise_sd, start_hand, dim=-1):
    S, B, D = obs_t.size()
    assert S == obs_tplus1.shape[0] and B == obs_tplus1.shape[1] and D == obs_tplus1.shape[2]
    assert D == 11

    policy_dim = 3

    with torch.no_grad():
        if dim==-1:
            x = torch.randint(0, 3, size=(B, 1))
        else:
            assert 0 <= dim <= 3
            x = torch.full(size=(B, 1), fill_value=dim)
        pi_t = F.one_hot(x, num_classes=policy_dim).detach().squeeze(1).unsqueeze(0).expand((S + 1, B, policy_dim))
        mask_pi_t = pi_t.repeat_interleave(3).view((S + 1, B, policy_dim * 3))
        finger_mask = torch.ones((S + 1, B, 2))
        mask_o_t = torch.ones((S + 1, B, D)) - torch.cat((mask_pi_t[:, :, :6], finger_mask, mask_pi_t[:, :, 6:]), 2)
        gaze_noise = (torch.randn((S + 1, B, D)) * gaze_noise_sd) * mask_o_t
        gaze_focus_noise = (torch.randn((S + 1, B, D)) * focus_noise_sd) * (1 -mask_o_t)
        overall_gaze_noise = gaze_noise + gaze_focus_noise
        if start_hand:
            overall_gaze_noise[0, :, 0:3] = 0.0
    return pi_t[0:S, :, :], obs_t + overall_gaze_noise[:S, :, :], obs_tplus1 + overall_gaze_noise[1:, :, :]

def add_gaze_based_noise_alternating(obs_t, obs_tplus1, gaze_noise_sd, focus_noise_sd, num_alternations, start_hand, dim=-1,):
    S, B, D = obs_t.size()
    assert S == obs_tplus1.shape[0] and B == obs_tplus1.shape[1] and D == obs_tplus1.shape[2]
    assert D == 11

    policy_dim = 3

    with torch.no_grad():
        if dim == -1:
            pis_all = torch.zeros((S+1, B, policy_dim))
            t_alt_gaze = random.sample(range(1, S), num_alternations)
            t_alt_gaze.sort()
            t_alt_gaze.append(S+1)
            last_t = 1
            for t_alt in t_alt_gaze:
                x = torch.randint(0, policy_dim, size=(B, 1))
                pi_t = F.one_hot(x, num_classes=policy_dim).detach().squeeze(1).unsqueeze(0).expand((t_alt-last_t, B, policy_dim))
                pis_all[last_t:t_alt, :, :] = pi_t
                last_t = t_alt
        else:
            x = torch.full(size=(B, 1), fill_value=dim)
            pis_all = F.one_hot(x, num_classes=policy_dim).detach().squeeze(1).unsqueeze(0).expand((S + 1, B, policy_dim))
        mask_pi_t = pis_all.repeat_interleave(3).view((S + 1, B, policy_dim * 3))
        finger_mask = torch.ones((S + 1, B, 2))
        mask_o_t = torch.ones((S + 1, B, D)) - torch.cat((mask_pi_t[:, :, :6], finger_mask, mask_pi_t[:, :, 6:]), 2)
        gaze_noise = (torch.randn((S + 1, B, D)) * gaze_noise_sd) * mask_o_t
        gaze_focus_noise = (torch.randn((S + 1, B, D)) * focus_noise_sd) * (1 -mask_o_t)
        overall_gaze_noise = gaze_noise + gaze_focus_noise
        if start_hand:
            overall_gaze_noise[0, :, 0:3] = 0.0
    return pis_all[1:, :, :], obs_t + overall_gaze_noise[:S, :, :], obs_tplus1 + overall_gaze_noise[1:, :, :]


def find_first_obj_gaze_onehot(pis):
    S, B, D = pis.shape
    obj_gaze_t = np.ones(B) * S
    for b in range(pis.shape[1]):
        for t in range(pis.shape[0]):
            if pis[t, b, 1] == 1:
                assert pis[t, b, 0] == 0 and pis[t, b, 2] == 0
                obj_gaze_t[b] = t
                break
    return obj_gaze_t

def find_first_goal_gaze_onehot(pis):
    S, B, D = pis.shape
    obj_gaze_t = np.ones(B) * S
    for b in range(pis.shape[1]):
        for t in range(pis.shape[0]):
            if pis[t, b, 2] == 1:
                assert pis[t, b, 0] == 0 and pis[t, b, 1] == 0
                obj_gaze_t[b] = t
                break
    return obj_gaze_t

def find_first_hand_gaze_onehot(pis):
    S, B, D = pis.shape
    obj_gaze_t = np.ones(B) * S
    for b in range(pis.shape[1]):
        for t in range(pis.shape[0]):
            if pis[t, b, 0] == 1:
                assert pis[t, b, 2] == 0 and pis[t, b, 1] == 0
                obj_gaze_t[b] = t
                break
    return obj_gaze_t