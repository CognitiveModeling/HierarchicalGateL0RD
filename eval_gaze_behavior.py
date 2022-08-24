import os
import numpy as np
import torch
import torch.nn.functional as F
import models.forward_inverse_model as FIM
import models.skip_net as SkipNet
from models.skip_data_gen import batch_to_skip_inputs
from dataloaders.fetch_event_dataset import create_event_dataloaders
from utils.gaze_modeling_utils import add_gaze_based_noise_alternating, find_first_goal_gaze_onehot, \
    find_first_hand_gaze_onehot, find_first_obj_gaze_onehot
from utils.torch_utils import compute_multivariate_normal_entropy


TORCH_SEED = 27


def main_gaze_experiment(params):

    print("Gaze experiment with params ", params)
    # result dir
    result_dir = params.gaze_experiment_dir
    os.makedirs(result_dir, exist_ok=True)

    # data related
    dataset_split_rs = params.dataset_split_rs
    num_data_per_dataset_train = 3200
    num_data_per_dataset_test = 3200
    seq_len = params.seq_len
    train_batch_size_per_dataset = 64
    val_batch_size_per_dataset = 3200

    # Load test dataloader and determine event boundary times
    _, _, test_dataloaders = create_event_dataloaders(
        seq_len=seq_len,
        dataset_split_rs=dataset_split_rs,
        num_data_train=num_data_per_dataset_train,
        num_data_test=num_data_per_dataset_test,
        train_batch_size=train_batch_size_per_dataset,
        test_batch_size=val_batch_size_per_dataset
    )
    grasp_ts = determine_event_boundary_times([test_dataloaders[0]])
    goal_ts = determine_event_boundary_goal_times([test_dataloaders[1]])

    rseeds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    checks = np.concatenate((np.arange(1, 301, 1), np.array([300])))

    mean_gaze_times_variance_hand_obj = np.zeros((len(rseeds), len(checks)))
    mean_gaze_times_variance_hand_goal = np.zeros((len(rseeds), len(checks)))
    mean_gaze_times_variance_hand_hand = np.zeros((len(rseeds), len(checks)))

    # Dimensionality of the scenario
    action_dim = 4
    state_dim = 11


    # Factor for scaling predictions
    output_factor = 0.1
    output_factor2 = 0.1

    # HPs from config
    latent_dim = params.latent_dim
    feature_dim = params.feature_dim
    num_layers_internal = params.num_layers_internal
    reg_lambda = params.reg_lambda
    postprocessing_layers = params.postprocessing_layers
    action_prediction_layers = params.action_prediction_layers
    warm_up_layers = params.warm_up_layers
    warm_up_inputs = params.warm_up_inputs
    gaussian_outputs =  True
    nll_beta = params.nll_beta
    rnn_type = 'GateL0RD'
    gaze_dim = 3

    # Skip dimensionality
    skip_input_dim = state_dim + latent_dim + gaze_dim
    skip_output_dim = state_dim


    # GRASPING:
    print("Experiment with reach-grasp-transport sequences")
    print("----------------------------------------------")
    for r, rs in enumerate(rseeds):
        rs_name = str(rs)
        for c, cp in enumerate(checks):
            print("Testing seed ", rs, " at validation epoch ", str(cp))

            check_name = '/checkpoint_v' + str(cp)

            # Gate net
            gate_net = FIM.ForwardInverseModel(
                obs_dim=state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                feature_dim=feature_dim,
                num_layers_internal=num_layers_internal,
                reg_lambda=reg_lambda,
                f_FM_layers=postprocessing_layers,
                f_IM_layers=action_prediction_layers,
                f_init_layers=warm_up_layers,
                f_init_inputs=warm_up_inputs,
                gauss_network=gaussian_outputs,
                nll_beta=nll_beta,
                rnn_type=rnn_type,
                additional_input_dim=gaze_dim
            )
            dir_name_gatel0rd_checkpoint =  params.model_dir  + '/' + rs_name + '/checkpoints/'  + check_name
            checkpoint = torch.load(dir_name_gatel0rd_checkpoint)
            gate_net.load_state_dict(checkpoint['gate_net_state_dict'])

            # Skip Network
            skip_net = SkipNet.SkipNet(
                input_dim=skip_input_dim,
                output_dim=skip_output_dim,
                feature_dim=params.skip_feature_dim,
                num_layers=params.skip_num_layers,
                gauss_net=params.skip_gauss_net,
                nll_beta=params.skip_nll_beta
            )
            dir_name_skip_checkpoint =  params.skip_model_dir  + '/' + rs_name + '/' + '/checkpoints/'  + check_name
            checkpoint = torch.load(dir_name_skip_checkpoint)
            skip_net.load_state_dict(checkpoint['skip_net_state_dict'])


            train_dataloaders, val_dataloaders, test_dataloaders = create_event_dataloaders(
                seq_len=seq_len,
                dataset_split_rs=dataset_split_rs,
                num_data_train=num_data_per_dataset_train,
                num_data_test=num_data_per_dataset_test,
                train_batch_size=train_batch_size_per_dataset,
                test_batch_size=val_batch_size_per_dataset
            )

            gaze_variance_hand = itertative_gaze_selection_gamma(
                dataloaders=[test_dataloaders[0]],
                network=gate_net, skip_network=skip_net,
                skip_end_t=25,
                seq_len=seq_len,
                batch_size=val_batch_size_per_dataset,
                start_steps=1,
                n_step_prediction=True,
                mode='observation',
                factor_output=output_factor,
                skip_predict_deltas=False,
                skip_output_dim=skip_output_dim,
                factor_output2=output_factor2,
                gaze_noise_sd=params.gaze_noise_sd,
                num_alternations=params.alt_gaze_num,
                start_hand=False,
                focus_noise_sd=params.focus_noise_sd,
                gamma=params.gaze_gamma
            )

            gaze_variance_hand_t = find_first_obj_gaze_onehot(gaze_variance_hand)
            gaze_variance_hand_t_diff = gaze_variance_hand_t - grasp_ts
            mean_gaze_times_variance_hand_obj[r, c] = np.mean(gaze_variance_hand_t_diff)

            gaze_variance_hand_t = find_first_goal_gaze_onehot(gaze_variance_hand)
            gaze_variance_hand_t_diff = gaze_variance_hand_t - grasp_ts
            mean_gaze_times_variance_hand_goal[r, c] = np.mean(gaze_variance_hand_t_diff)

            gaze_variance_hand_t = find_first_hand_gaze_onehot(gaze_variance_hand)
            gaze_variance_hand_t_diff = gaze_variance_hand_t - grasp_ts
            mean_gaze_times_variance_hand_hand[r, c] = np.mean(gaze_variance_hand_t_diff)

    np.save(result_dir + 'grasp_gaze_' + str(params.gaze_gamma) + 'obj.npy', mean_gaze_times_variance_hand_obj)
    np.save(result_dir + 'grasp_gaze_' + str(params.gaze_gamma) + 'hand.npy', mean_gaze_times_variance_hand_hand)
    np.save(result_dir + 'grasp_gaze_' + str(params.gaze_gamma) + 'goal.npy', mean_gaze_times_variance_hand_goal)

    # POINTING:
    print("Experiment with pointing sequences")
    print("----------------------------------------------")
    for r, rs in enumerate(rseeds):
        rs_name = str(rs)
        for c, cp in enumerate(checks):
            print("Testing seed ", rs, " at validation epoch ", str(cp))

            check_name = '/checkpoint_v' + str(cp)

            # Gate net
            gate_net = FIM.ForwardInverseModel(
                obs_dim=state_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                feature_dim=feature_dim,
                num_layers_internal=num_layers_internal,
                reg_lambda=reg_lambda,
                f_FM_layers=postprocessing_layers,
                f_IM_layers=action_prediction_layers,
                f_init_layers=warm_up_layers,
                f_init_inputs=warm_up_inputs,
                gauss_network=gaussian_outputs,
                nll_beta=nll_beta,
                rnn_type=rnn_type,
                additional_input_dim=gaze_dim
            )
            dir_name_gatel0rd_checkpoint = params.model_dir + '/' + rs_name + '/checkpoints/' + check_name
            checkpoint = torch.load(dir_name_gatel0rd_checkpoint)
            gate_net.load_state_dict(checkpoint['gate_net_state_dict'])

            # Skip Network
            skip_net = SkipNet.SkipNet(
                input_dim=skip_input_dim,
                output_dim=skip_output_dim,
                feature_dim=params.skip_feature_dim,
                num_layers=params.skip_num_layers,
                gauss_net=params.skip_gauss_net,
                nll_beta=params.skip_nll_beta
            )
            dir_name_skip_checkpoint = params.skip_model_dir + '/' + rs_name + '/' + '/checkpoints/' + check_name
            checkpoint = torch.load(dir_name_skip_checkpoint)
            skip_net.load_state_dict(checkpoint['skip_net_state_dict'])

            train_dataloaders, val_dataloaders, test_dataloaders = create_event_dataloaders(
                seq_len=seq_len,
                dataset_split_rs=dataset_split_rs,
                num_data_train=num_data_per_dataset_train,
                num_data_test=num_data_per_dataset_test,
                train_batch_size=train_batch_size_per_dataset,
                test_batch_size=val_batch_size_per_dataset
            )

            gaze_variance_hand = itertative_gaze_selection_gamma(
                dataloaders=[test_dataloaders[1]],
                network=gate_net, skip_network=skip_net,
                skip_end_t=25,
                seq_len=seq_len,
                batch_size=val_batch_size_per_dataset,
                start_steps=1,
                n_step_prediction=True,
                mode='observation',
                factor_output=output_factor,
                skip_predict_deltas=False,
                skip_output_dim=skip_output_dim,
                factor_output2=output_factor2,
                gaze_noise_sd=params.gaze_noise_sd,
                num_alternations=params.alt_gaze_num,
                start_hand=False,
                focus_noise_sd=params.focus_noise_sd,
                gamma=params.gaze_gamma
            )

            gaze_variance_hand_t = find_first_obj_gaze_onehot(gaze_variance_hand)
            gaze_variance_hand_t_diff = mean_diff_interleave(gaze_variance_hand_t, goal_ts, 1000, -1)
            mean_gaze_times_variance_hand_obj[r, c] = np.mean(gaze_variance_hand_t_diff)

            gaze_variance_hand_t = find_first_goal_gaze_onehot(gaze_variance_hand)
            gaze_variance_hand_t_diff = mean_diff_interleave(gaze_variance_hand_t, goal_ts, 1000, -1)
            mean_gaze_times_variance_hand_goal[r, c] = np.mean(gaze_variance_hand_t_diff)

            gaze_variance_hand_t = find_first_hand_gaze_onehot(gaze_variance_hand)
            gaze_variance_hand_t_diff = mean_diff_interleave(gaze_variance_hand_t, goal_ts, 1000, -1)
            mean_gaze_times_variance_hand_hand[r, c] = np.mean(gaze_variance_hand_t_diff)

    np.save(result_dir + 'point_gaze_' + str(params.gaze_gamma) + 'obj.npy', mean_gaze_times_variance_hand_obj)
    np.save(result_dir + 'point_gaze_' + str(params.gaze_gamma) + 'hand.npy', mean_gaze_times_variance_hand_hand)
    np.save(result_dir + 'point_gaze_' + str(params.gaze_gamma) + 'goal.npy', mean_gaze_times_variance_hand_goal)


def gaze_selection_batch(values0, values1, values2):
    all_values = np.stack([values0, values1, values2], -1)
    pis = np.argmin(all_values, axis=1)
    return pis

def mean_diff_interleave(a, b, a_mask_value, b_mask_value):
    # Take the diff between a and b except if b==mask_value
    res = []
    for i,j in zip(a, b):
        if i != a_mask_value and j != b_mask_value:
            res.append(i-j)
    return sum(res)/len(res)

def determine_event_boundary_times(dataloaders):
    torch.random.manual_seed(TORCH_SEED)
    ts = []
    with torch.no_grad():
        for data in zip(*dataloaders):
            list_obs = []
            for d in range(1):
                data_d = data[d]
                list_obs.append(data_d[0].permute(1, 0, 2))
            obs = torch.cat(list_obs, dim=1).float()

            for b in range(obs.shape[1]):
                for t in range(24):
                    if np.linalg.norm(obs[t, b, [3, 4, 5]] - obs[t, b, [0, 1, 2]]) < 0.02:
                        ts.append(t)
                        break
    return np.array(ts)

def determine_event_boundary_goal_times(dataloaders):
    torch.random.manual_seed(TORCH_SEED)
    ts = []
    with torch.no_grad():
        for data in zip(*dataloaders):
            list_obs = []
            for d in range(1):
                data_d = data[d]
                list_obs.append(data_d[0].permute(1, 0, 2))
            obs = torch.cat(list_obs, dim=1).float()
            for b in range(obs.shape[1]):
                goal_reached = False
                for t in range(24):
                    if np.linalg.norm(obs[t, b, [8, 9, 10]] - obs[t, b, [0, 1, 2]]) < 0.02:
                        ts.append(t)
                        goal_reached = True
                        break
                if not goal_reached:
                    # Append masking value if goal is never reached
                    ts.append(-1)
    return np.array(ts)


def uncertainty_of_skip_gamma(
        dataloaders,
        network,
        skip_network,
        skip_t,
        seq_len,
        batch_size,
        start_steps,
        n_step_prediction,
        mode,
        factor_output,
        factor_output2,
        skip_predict_deltas,
        skip_output_dim,
        gaze_noise_sd=-1,
        gaze_dim=-1,
        num_alternations=3,
        start_hand=False,
        focus_noise_sd=0.0,
        previous_pis=None,
        gamma=0.5
):

    network = network.eval()
    skip_network = skip_network.eval()

    # 1step vs N step prediction?
    val_ss = np.ones((skip_t, batch_size))
    if n_step_prediction:
        val_ss[start_steps:skip_t, :] = -1

    val_count = 0
    num_datasets = len(dataloaders)

    # Dim definitions:
    hand_dims = [0, 1, 2]
    obj_dims = [3, 4, 5]
    goal_dims = [8, 9, 10]

    mean_variance = []
    mean_entropy = []
    mean_variance_hand = []
    mean_variance_obj = []
    mean_variance_goal = []

    with torch.no_grad():
        for data in zip(*dataloaders):
            list_obs = []
            list_actions = []
            list_delta_obs = []
            for d in range(num_datasets):
                data_d = data[d]
                list_obs.append(data_d[0].permute(1, 0, 2))
                list_actions.append(data_d[1].permute(1, 0, 2))
                list_delta_obs.append(data_d[2].permute(1, 0, 2))
            obs = torch.cat(list_obs, dim=1).float()
            actions = torch.cat(list_actions, dim=1).float()
            next_obs = obs + torch.cat(list_delta_obs, dim=1).float()

            pis, obs, next_obs = add_gaze_based_noise_alternating(
                obs,
                next_obs,
                gaze_noise_sd,
                dim=gaze_dim,
                num_alternations=num_alternations,
                start_hand=start_hand,
                focus_noise_sd=focus_noise_sd
            )

            pis = pis.clone()
            if previous_pis is not None:
                pis[:(skip_t - 1), :, :] = previous_pis[:(skip_t - 1), :, :].clone()

            s, b, _ = obs.shape

            add_info = pis[:skip_t, :, :]
            _, z, gate_reg, _, _, _, ll_sigmas, _ = network.forward_n_step(
                obs_batch=obs[:skip_t, :, :],
                train_schedule=val_ss,
                predict_o_deltas=True,
                factor_o_delta=factor_output,
                action_batch=actions[:skip_t, :, :],
                mode=mode,
                predict_a_deltas=False,
                factor_a_delta=factor_output2,
                additional_info=add_info
            )

            act_input = pis

            skip_inps = batch_to_skip_inputs(
                xs=obs[skip_t - 1, :, :],
                hs=z[0, skip_t - 1, :, :],
                add_inps=act_input[skip_t - 1, :, :]
            )
            skip_ys, skip_sigmas = skip_network.forward(skip_inps)
            skip_outs = skip_ys
            if skip_predict_deltas:
                skip_outs[:, :skip_output_dim] += skip_inps[:, :skip_output_dim]

            mean_variance += (ll_sigmas[-1, :, :].mean(dim=1).detach().numpy().tolist())
            mean_variance_hand += (((1 - gamma) * skip_sigmas[:, hand_dims].mean(dim=1) + gamma * ll_sigmas[-1, :,
                                                                                                  hand_dims].mean(
                dim=1)).detach().numpy().tolist())
            mean_variance_obj += (((1 - gamma) * skip_sigmas[:, obj_dims].mean(dim=1) + gamma * ll_sigmas[-1, :,
                                                                                                obj_dims].mean(
                dim=1)).detach().numpy().tolist())
            mean_variance_goal += (((1 - gamma) * skip_sigmas[:, goal_dims].mean(dim=1) + gamma * ll_sigmas[-1, :,
                                                                                                  goal_dims].mean(
                dim=1)).detach().numpy().tolist())
            mean_entropy += (compute_multivariate_normal_entropy(skip_ys, skip_sigmas).detach().numpy().tolist())

            val_count += 1

    return mean_variance, mean_variance_hand, mean_variance_obj, mean_variance_goal, mean_entropy


def itertative_gaze_selection_gamma(
        dataloaders,
        network,
        skip_network,
        skip_end_t,
        seq_len,
        batch_size,
        start_steps,
        n_step_prediction,
        mode,
        factor_output,
        factor_output2,
        skip_predict_deltas,
        skip_output_dim,
        gaze_noise_sd=-1,
        num_alternations=3,
        start_hand=False,
        focus_noise_sd=0.0,
        gamma=0.5
):

    previous_gaze = None
    with torch.no_grad():
        for skip_t in range(1, skip_end_t + 1):
            val_variances_hand_per_gaze = []
            for gdim in range(3):
                torch.random.manual_seed(TORCH_SEED) # always set same seed
                val_variance, val_variance_hand, val_variance_obj, val_variance_goal, val_entropy = uncertainty_of_skip_gamma(
                    dataloaders=dataloaders,
                    network=network,
                    skip_network=skip_network,
                    skip_t=skip_t,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    start_steps=start_steps,
                    n_step_prediction=n_step_prediction,
                    mode=mode,
                    factor_output=factor_output,
                    skip_predict_deltas=skip_predict_deltas,
                    skip_output_dim=skip_output_dim,
                    factor_output2=factor_output2,
                    gaze_noise_sd=gaze_noise_sd,
                    gaze_dim=gdim,
                    num_alternations=num_alternations,
                    gamma=gamma,
                    start_hand=start_hand,
                    focus_noise_sd=focus_noise_sd,
                    previous_pis=previous_gaze
                )
                val_variances_hand_per_gaze.append(val_variance_hand)

            pis_t = gaze_selection_batch(
                val_variances_hand_per_gaze[0],
                val_variances_hand_per_gaze[1],
                val_variances_hand_per_gaze[2]
            )
            pis_one_hot = F.one_hot(
                torch.from_numpy(pis_t).to(torch.int64),
                num_classes=3
            ).detach().squeeze(1).unsqueeze(0)
            if previous_gaze is None:
                previous_gaze = pis_one_hot
            else:
                previous_gaze = torch.cat((previous_gaze, pis_one_hot), 0)

    return previous_gaze