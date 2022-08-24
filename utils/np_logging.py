import os
import numpy as np

def save_val_arrays(target_dir, name, x_obs, x_act, x_gating, type='MSE'):


    obs_np_file = os.path.join(target_dir, name + "_obs_" + type + "_np")
    np.save(obs_np_file, x_obs)

    act_np_file = os.path.join(target_dir, name + "_act_" + type + "_np")
    np.save(act_np_file, x_act)

    gate_np_file = os.path.join(target_dir, name + "_gating_np")
    np.save(gate_np_file, x_gating)


def load_val_arrays(target_dir, name, x_obs, x_act, x_gating, val_num, type='MSE'):

    obs_np_file = os.path.join(target_dir, name + "_obs_" + type + "_np.npy")
    x_obs[:, :val_num] = np.load(obs_np_file)[:, :val_num]

    act_np_file = os.path.join(target_dir, name + "_act_" + type + "_np.npy")
    x_act[:, :val_num] = np.load(act_np_file)[:, :val_num]

    gate_np_file = os.path.join(target_dir, name + "_gating_np.npy")
    x_gating[:, :val_num] = np.load(gate_np_file)[:, :val_num]

    return x_obs, x_act, x_gating