import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import models.forward_inverse_model as FIM
import utils.colordef as colors

from dataloaders.fetch_event_dataset import create_event_dataloaders, create_apex_dataloaders
from utils.os_managment import create_res_dirs
from utils.gating_analysis import calc_gate_rate, calc_gate_dims_used, calc_gate_open_times
from utils.set_rs import set_rs
from utils.np_logging import load_val_arrays, save_val_arrays
from utils.scheduled_sampling_helper import exponential_ss_prob
from utils.gaze_modeling_utils import add_gaze_based_noise_alternating



def main_FPP_events(params):

    print("Starting FPP Forward-Inverse model training with params ", params)

    rand_seed = params.rs
    set_rs(rand_seed)

    # create directories for logging
    target_dir = params.model_dir + '/' + str(rand_seed) + '/'
    checkpoint_dir, plot_dir, metrics_dir = create_res_dirs(target_dir)


    # dimensionality of the scenario
    action_dim = 4
    state_dim = 11

    # factors for scaling predictions
    output_factor = 0.1 # observation prediction
    output_factor2 = 0.1 # action prediction

    # gauss networks?
    gaussian_outputs = params.gauss_net

    # gaze related
    add_gaze = False
    gaze_dim= 0
    if params.add_gaze:
        add_gaze = True
        gaze_dim = 3
    gaze_noise_sd = params.gaze_noise_sd
    focus_noise_sd = params.focus_noise_sd
    gaze_alternations = params.alt_gaze_num
    start_hand = params.start_hand


    # create the low-level network
    gate_net = FIM.ForwardInverseModel(
        reg_lambda= params.reg_lambda,
        obs_dim=state_dim,
        action_dim=action_dim,
        latent_dim=params.latent_dim,
        feature_dim=params.feature_dim,
        num_layers_internal=params.num_layers_internal,
        f_FM_layers=params.postprocessing_layers,
        f_IM_layers=params.action_prediction_layers,
        f_init_layers=params.warm_up_layers,
        f_init_inputs=params.warm_up_inputs,
        gauss_network=gaussian_outputs,
        nll_beta=params.nll_beta,
        rnn_type=params.rnn_type,
        additional_input_dim=gaze_dim
    )

    gate_optimizer = gate_net.get_optimizer(params.lr)
    ss_slope = params.ss_slope

    # train using teacher forcing?
    if params.train_mode == 'TF':
        teacher_force_training = True
        gatel0rd_train_mode = 'observation'
    else:
        teacher_force_training = False
        gatel0rd_train_mode = params.train_mode

    # gradient norm clipping?
    grad_clip = True
    grad_clip_value = params.grad_clip_value
    if grad_clip_value <= 0:
        grad_clip = False


    # data related
    dataset_split_rs = params.dataset_split_rs
    num_data_per_dataset_train = 3200
    num_data_per_dataset_test = 3200
    b_size_per_dataset = 64 #32
    start_steps = 1  # Number of inputs always provided
    seq_len = params.seq_len

    if params.apex_data:
        # Triple every number
        num_data_per_dataset_train = int(num_data_per_dataset_train*3)
        num_data_per_dataset_test = int(num_data_per_dataset_test)
        train_batch_size_per_dataset = int(b_size_per_dataset*3)
        val_batch_size_per_dataset = int(b_size_per_dataset * 10)
        train_dataloaders, val_dataloaders, test_dataloaders = create_apex_dataloaders(
            seq_len=seq_len,
            dataset_split_rs=dataset_split_rs,
            num_data_train=num_data_per_dataset_train,
            num_data_test=num_data_per_dataset_test,
            val_batch_size=val_batch_size_per_dataset,
            train_batch_size=train_batch_size_per_dataset
        )
        val_dataset_names = ['grasp', 'full']
    else:
        val_batch_size_per_dataset = int(b_size_per_dataset * 10)
        train_batch_size_per_dataset = b_size_per_dataset
        train_dataloaders, val_dataloaders, test_dataloaders = create_event_dataloaders(
            seq_len=seq_len,
            dataset_split_rs=dataset_split_rs,
            num_data_train=num_data_per_dataset_train,
            num_data_test=num_data_per_dataset_test,
            train_batch_size=train_batch_size_per_dataset,
            test_batch_size=val_batch_size_per_dataset
        )
        val_dataset_names = ['transport', 'point', 'stretch']

    num_datasets = len(val_dataloaders)
    num_train_datasets = len(train_dataloaders)
    overall_batch_size = int(val_batch_size_per_dataset)
    overall_train_batch_size = int(train_batch_size_per_dataset * num_train_datasets)

    # time intervals for evaluation, plotting & saving metrics
    t_validation = 100
    t_plots = 20
    t_save_metrics = 10

    num_validations = 10 * params.train_len
    num_epochs = t_validation * num_validations


    # metrics, managed as simple numpy arrays

    # validation metrics
    val_1step_obs_MSE_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    val_1step_act_MSE_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    val_1step_obs_NLL_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    val_1step_act_NLL_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    val_1step_gating_over_t = np.zeros((num_datasets, 3, num_validations), dtype='float64')
    val_gaze_variance_over_t = np.zeros((num_datasets, 3, state_dim, num_validations), dtype='float64')
    
    # testing metrics
    test_1step_obs_MSE_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    test_1step_act_MSE_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    test_1step_obs_NLL_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    test_1step_act_NLL_over_t = np.zeros((num_datasets, num_validations), dtype='float64')
    test_1step_gating_over_t = np.zeros((num_datasets, 3, num_validations), dtype='float64')
    test_gaze_variance_over_t = np.zeros((num_datasets, 3, state_dim, num_validations), dtype='float64')

    # train metrics
    loss_gate_net = np.zeros(num_epochs, dtype='float64')

    # counters
    epoch_start = 0
    validations = 0

    # checkpoint name
    fname = os.path.join(checkpoint_dir, "checkpoint")
    if os.path.isfile(fname):

        # load existing model if available
        dir_name_checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(dir_name_checkpoint)
        gate_net.load_state_dict(checkpoint['gate_net_state_dict'])
        gate_optimizer.load_state_dict(checkpoint['gate_optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        validations = checkpoint['validations']

        # load existing metrics
        val_1step_obs_MSE_over_t, val_1step_act_MSE_over_t, val_1step_gating_over_t = load_val_arrays(
            metrics_dir,
            "val_1step",
            val_1step_obs_MSE_over_t,
            val_1step_act_MSE_over_t,
            val_1step_gating_over_t,
            validations, 'MSE'
        )
        test_1step_obs_MSE_over_t, test_1step_act_MSE_over_t, test_1step_gating_over_t = load_val_arrays(
            metrics_dir,
            "test_1step",
            test_1step_obs_MSE_over_t,
            test_1step_act_MSE_over_t,
            test_1step_gating_over_t,
            validations,
            'MSE'
        )

        if gaussian_outputs:
            val_1step_obs_NLL_over_t, val_1step_act_NLL_over_t, _ = load_val_arrays(
                metrics_dir,
                "val_1step",
                val_1step_obs_NLL_over_t,
                val_1step_act_NLL_over_t,
                val_1step_gating_over_t,
                validations,
                'NLL'
            )
            test_1step_obs_NLL_over_t, test_1step_act_NLL_over_t, _ = load_val_arrays(
                metrics_dir,
                "test_1step",
                test_1step_obs_NLL_over_t,
                test_1step_act_NLL_over_t,
                test_1step_gating_over_t,
                validations,
                'NLL'
            )

        loss_gate_net_file = os.path.join(metrics_dir, "loss_gate_net_np.npy")
        loss_gate_net[:epoch_start] = np.load(loss_gate_net_file)[:epoch_start]

        gaze_variance_file = os.path.join(metrics_dir, "val_gaze_variance_over_t_np.npy")
        val_gaze_variance_over_t[:, :, :, :epoch_start] = np.load(gaze_variance_file)[:, :, :, :epoch_start]

        gaze_variance_file = os.path.join(metrics_dir, "test_gaze_variance_over_t_np.npy")
        test_gaze_variance_over_t[:, :, :, :epoch_start] = np.load(gaze_variance_file)[:, :, :, :epoch_start]

    epoch = epoch_start
    print("------------ ")
    for epoch in range(epoch_start, num_epochs):

        if epoch%t_validation == 0:

            print("Evaluation ", validations, " after epoch ", epoch)

            for num_ds in range(num_datasets):

                # VALIDATION:
                # 1. 1step:
                val_1step_obs_MSE_over_t[num_ds, validations], val_1step_act_MSE_over_t[num_ds, validations], \
                val_1step_gating_over_t[num_ds, 0, validations], val_1step_gating_over_t[num_ds, 1, validations], \
                val_1step_gating_over_t[num_ds, 2, validations], val_1step_obs_NLL_over_t[num_ds, validations], \
                val_1step_act_NLL_over_t[num_ds, validations], _ = eval_fc(
                    dataloaders=[val_dataloaders[num_ds]],
                    network=gate_net,
                    seq_len=seq_len,
                    batch_size=overall_batch_size,
                    start_steps=start_steps,
                    n_step_prediction=False,
                    mode='observation',
                    factor_output=output_factor,
                    example_plot=(validations%t_plots == 0),
                    example_plot_directory_name=plot_dir + '/plot_val_' + str(validations)
                                                + '_1step_' + val_dataset_names[num_ds],
                    factor_output2=output_factor2,
                    gauss_outputs=gaussian_outputs,
                    add_gaze=add_gaze,
                    gaze_noise_sd=gaze_noise_sd,
                    focus_noise_sd=focus_noise_sd,
                    num_alternations=gaze_alternations,
                    start_hand=start_hand
                )

                if add_gaze:
                    # additional evaluation for gaze modeling
                    for d in range(3):
                        _, _, _, _, _, _, _, val_gaze_variance_over_t[num_ds, d, :, validations] =eval_fc(
                            dataloaders=[val_dataloaders[num_ds]],
                            network=gate_net,
                            seq_len=seq_len,
                            batch_size=overall_batch_size,
                            start_steps=start_steps,
                            n_step_prediction=False,
                            mode='observation',
                            factor_output=output_factor,
                            example_plot=(validations % t_plots == 0),
                            example_plot_directory_name= plot_dir
                                                         + '/plot_val_gazedim' + str(d)+ '_' + str(validations)
                                                         + '_1step_' + val_dataset_names[num_ds],
                            factor_output2=output_factor2,
                            gauss_outputs=gaussian_outputs,
                            add_gaze=add_gaze,
                            gaze_noise_sd=gaze_noise_sd,
                            focus_noise_sd=focus_noise_sd,
                            gaze_dim=d,
                            start_hand=start_hand
                        )


                # TESTING:
                test_1step_obs_MSE_over_t[num_ds, validations], test_1step_act_MSE_over_t[num_ds, validations], \
                test_1step_gating_over_t[num_ds, 0, validations], test_1step_gating_over_t[num_ds, 1, validations], \
                test_1step_gating_over_t[num_ds, 2, validations], test_1step_obs_NLL_over_t[num_ds, validations], \
                test_1step_act_NLL_over_t[num_ds, validations], _ = eval_fc(
                    dataloaders=[test_dataloaders[num_ds]],
                    network=gate_net,
                    seq_len=seq_len,
                    batch_size=overall_batch_size,
                    start_steps=start_steps,
                    n_step_prediction=False,
                    mode='observation',
                    factor_output=output_factor,
                    example_plot=False,
                    factor_output2=output_factor2,
                    gauss_outputs=gaussian_outputs,
                    add_gaze=add_gaze,
                    gaze_noise_sd=gaze_noise_sd,
                    focus_noise_sd=focus_noise_sd,
                    num_alternations=gaze_alternations,
                    start_hand=start_hand
                )

                if add_gaze:
                    # Additional evaluation for gaze modeling
                    for d in range(3):
                        _, _, _, _, _, _, _, test_gaze_variance_over_t[num_ds, d, :, validations] = eval_fc(
                            dataloaders=[test_dataloaders[num_ds]],
                            network=gate_net,
                            seq_len=seq_len,
                            batch_size=overall_batch_size,
                            start_steps=start_steps,
                            n_step_prediction=False,
                            mode='observation',
                            factor_output=output_factor,
                            example_plot=False,
                            factor_output2=output_factor2,
                            gauss_outputs=gaussian_outputs,
                            add_gaze=add_gaze,
                            gaze_noise_sd=gaze_noise_sd,
                            focus_noise_sd=focus_noise_sd,
                            gaze_dim=d,
                            start_hand=start_hand
                        )

            if validations % t_save_metrics == 0:
                print("Saving metrics ...")
                # Save all metrics:
                save_val_arrays(
                    metrics_dir,
                    "val_1step",
                    val_1step_obs_MSE_over_t,
                    val_1step_act_MSE_over_t,
                    val_1step_gating_over_t,
                    "MSE"
                )
                save_val_arrays(
                    metrics_dir,
                    "val_1step",
                    val_1step_obs_NLL_over_t,
                    val_1step_act_NLL_over_t,
                    val_1step_gating_over_t,
                    "NLL"
                )
                save_val_arrays(
                    metrics_dir,
                    "test_1step",
                    test_1step_obs_MSE_over_t,
                    test_1step_act_MSE_over_t,
                    test_1step_gating_over_t,
                    "MSE"
                )
                save_val_arrays(
                    metrics_dir,
                    "test_1step",
                    test_1step_obs_NLL_over_t,
                    test_1step_act_NLL_over_t,
                    test_1step_gating_over_t,
                    "NLL"
                )

                loss_gate_net_file = os.path.join(metrics_dir, "loss_gate_net_np.npy")
                np.save(loss_gate_net_file, loss_gate_net)
                gaze_variance_file = os.path.join(metrics_dir, "val_gaze_variance_over_t_np.npy")
                np.save(gaze_variance_file, val_gaze_variance_over_t)
                gaze_variance_file = os.path.join(metrics_dir, "test_gaze_variance_over_t_np.npy")
                np.save(gaze_variance_file, test_gaze_variance_over_t)

            validations += 1

            # checkpointing
            for checkpoint_name in ["checkpoint", "checkpoint_v" + str(validations)]:
                dir_name_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
                # Save a checkpoint
                torch.save({
                    'epoch': epoch,
                    'gate_net_state_dict': gate_net.state_dict(),
                    'gate_optimizer_state_dict': gate_optimizer.state_dict(),
                    'validations': validations
                }, dir_name_checkpoint)


        if validations < num_validations:
            # TRAINING

            gate_net = gate_net.train()

            # scheduled sampling?
            ss = np.ones((seq_len, overall_train_batch_size))
            if ss_slope == 0:
                ss_epsilon = 0.0
            else:
                ss_epsilon = exponential_ss_prob(epoch=epoch, slope=ss_slope, min_value=0.05)
            if teacher_force_training:
                ss_epsilon = 1.0
            ss[start_steps:seq_len, :] = ss_epsilon


            loss_gate_net_sum = 0.0
            train_count = 0

            for data in zip(*train_dataloaders):
                list_obs = []
                list_actions = []
                list_delta_obs = []
                for d in range(num_train_datasets):
                    data_d = data[d]
                    list_obs.append(data_d[0].permute(1, 0, 2))
                    list_actions.append(data_d[1].permute(1, 0, 2))
                    list_delta_obs.append(data_d[2].permute(1, 0, 2))
                obs = torch.cat(list_obs, dim=1).float()
                actions = torch.cat(list_actions, dim=1).float()
                delta_obs = torch.cat(list_delta_obs, dim=1).float()

                s, b, _ = obs.shape

                next_obs = obs + delta_obs
                next_actions = actions[1:, :, :]
                next_obs_target = next_obs.clone()
                gaze_policy = None

                if add_gaze:
                    gaze_policy, obs, next_obs = add_gaze_based_noise_alternating(
                        obs,
                        next_obs,
                        gaze_noise_sd,
                        focus_noise_sd=focus_noise_sd,
                        num_alternations=gaze_alternations,
                        start_hand=start_hand
                    )

                gate_net.zero_grad()
                gate_optimizer.zero_grad()
                y_obs, z, gate_reg, _, _, y_act, sigma_obs, sigma_act = gate_net.forward_n_step(
                    obs_batch=obs,
                    train_schedule=ss,
                    predict_o_deltas=True,
                    factor_o_delta=output_factor,
                    action_batch=actions,
                    mode=gatel0rd_train_mode,
                    predict_a_deltas=False,
                    factor_a_delta=output_factor2,
                    additional_info=gaze_policy
                )
                act_hat = y_act[:(seq_len -1), :, :]
                sigma_act_hat = None
                if gaussian_outputs:
                    sigma_act_hat = sigma_act[:(seq_len -1), :, :]

                loss = gate_net.loss(
                    out=y_obs,
                    tar=next_obs_target,
                    g_regs=gate_reg,
                    out2=act_hat,
                    tar2=next_actions,
                    sigmas=sigma_obs,
                    sigmas2=sigma_act_hat
                )
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(gate_net.parameters(), grad_clip_value)
                gate_optimizer.step()

                loss_gate_net_sum += loss.detach().item()
                train_count += 1
            loss_gate_net[epoch] = loss_gate_net_sum/train_count

            print("Training epoch ", epoch, "/", num_epochs, " done, mean loss = ", loss_gate_net_sum/train_count)


    # Save everything (twice)
    final_checkpoint_names = ["checkpoint",  "final_checkpoint"]
    for fname in final_checkpoint_names:
        dir_name_checkpoint = os.path.join(checkpoint_dir, fname)
        torch.save({
            'epoch': epoch + 1,
            'gate_net_state_dict': gate_net.state_dict(),
            'gate_optimizer_state_dict': gate_optimizer.state_dict(),
            'validations': validations
        }, dir_name_checkpoint)

    # Save all metrics:
    save_val_arrays(
        metrics_dir,
        "val_1step",
        val_1step_obs_MSE_over_t,
        val_1step_act_MSE_over_t,
        val_1step_gating_over_t,
        "MSE"
    )
    save_val_arrays(
        metrics_dir,
        "val_1step",
        val_1step_obs_NLL_over_t,
        val_1step_act_NLL_over_t,
        val_1step_gating_over_t,
        "NLL"
    )

    save_val_arrays(
        metrics_dir,
        "test_1step",
        test_1step_obs_MSE_over_t,
        test_1step_act_MSE_over_t,
        test_1step_gating_over_t,
        "MSE"
    )
    save_val_arrays(
        metrics_dir,
        "test_1step",
        test_1step_obs_NLL_over_t,
        test_1step_act_NLL_over_t,
        test_1step_gating_over_t,
        "NLL"
    )

    loss_gate_net_file = os.path.join(metrics_dir, "loss_gate_net_np.npy")
    np.save(loss_gate_net_file, loss_gate_net)
    gaze_variance_file = os.path.join(metrics_dir, "val_gaze_variance_over_t_np.npy")
    np.save(gaze_variance_file, val_gaze_variance_over_t)
    gaze_variance_file = os.path.join(metrics_dir, "test_gaze_variance_over_t_np.npy")
    np.save(gaze_variance_file, test_gaze_variance_over_t)


def eval_fc(
        dataloaders,
        network,
        seq_len,
        batch_size,
        start_steps,
        n_step_prediction,
        mode,
        factor_output,
        factor_output2,
        example_plot=False,
        example_plot_directory_name='',
        gauss_outputs=False,
        add_gaze=False,
        focus_noise_sd=0.0,
        gaze_noise_sd=0.1,
        gaze_dim=-1,
        num_alternations=3,
        start_hand=False,
        obs_dim=11,
):
    network = network.eval()

    # 1-step vs N-step prediction?
    val_ss = np.ones((seq_len, batch_size))
    if n_step_prediction:
        val_ss[start_steps:seq_len, :] = -1

    # MSE measures
    val_obs_MSE_sum = 0.0
    val_act_MSE_sum = 0.0

    # NLL measures
    val_obs_NLL_sum = 0.0
    val_act_NLL_sum = 0.0

    # gating metrics
    val_gate_rate_sum = 0.0
    val_gate_dims_sum = 0.0
    val_gate_times_sum = 0.0

    # average variances
    mean_variances_obs = np.zeros((obs_dim))

    val_count = 0
    num_datasets = len(dataloaders)

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
            delta_obs = torch.cat(list_delta_obs, dim=1).float()

            s, b, _ = obs.shape
            next_obs = obs + delta_obs
            next_actions = actions[1:, :, :]
            next_obs_target = next_obs.clone()
            gaze_policy = None
            if add_gaze:
                gaze_policy, obs, next_obs = add_gaze_based_noise_alternating(
                    obs,
                    next_obs,
                    gaze_noise_sd,
                    focus_noise_sd=focus_noise_sd,
                    num_alternations=num_alternations,
                    start_hand=start_hand,
                    dim=gaze_dim
                )

            y_obs, z, gate_reg, _, _, y_act, sigma_obs, sigma_act = network.forward_n_step(
                obs_batch=obs,
                train_schedule=val_ss,
                predict_o_deltas=True,
                factor_o_delta=factor_output,
                action_batch=actions,
                mode=mode,
                predict_a_deltas=False,
                factor_a_delta=factor_output2,
                additional_info=gaze_policy
            )

            # MSE measure updates
            val_obs_MSE_sum += network.MSE(y_obs, next_obs_target).detach().item()
            val_act_MSE_sum += network.MSE(y_act[:(seq_len - 1), :, :], next_actions).detach().item()

            if gauss_outputs:
                # NLL analysis
                val_obs_NLL_sum += network.NLL(
                    y_obs,
                    sigma_obs,
                    next_obs_target,
                    ignore_beta=True
                ).detach().item()
                val_act_NLL_sum += network.NLL(
                    y_act[:(seq_len - 1), :, :],
                    sigma_act[:(seq_len - 1), :, :],
                    next_actions,
                    ignore_beta=True
                ).detach().item()
                mean_variances_obs += np.mean(np.mean(sigma_obs.detach().numpy(), 0), 0)

            gate_activity = gate_reg[0, :, :, :]
            val_gate_rate_sum += calc_gate_rate(gate_activity)
            val_gate_dims_sum += calc_gate_dims_used(gate_activity)
            val_gate_times_sum += calc_gate_open_times(gate_activity)

            if val_count == 0 and example_plot:

                # create an example plot for first validation
                y_act_np = y_act.detach().numpy()
                sigma_obs_np = None
                sigma_act_np = None
                if gauss_outputs:
                    sigma_obs_np = sigma_obs.detach().numpy()
                    sigma_act_np = sigma_act.detach().numpy()
                data_gaze_policy_np = None
                if gaze_policy is not None:
                    data_gaze_policy_np = gaze_policy.detach().numpy()

                # One plot for a transportation sequence
                reach_sim = random.randint(0, 10)
                log_one_plot(
                    data_y=y_obs.detach().numpy(),
                    data_target=next_obs_target.detach().numpy(),
                    data_z=z.detach().numpy(),
                    data_actions=y_act_np,
                    data_target_actions=next_actions.detach().numpy(),
                    directory_name=example_plot_directory_name,
                    sim=reach_sim,
                    data_g=gate_activity.detach().numpy(),
                    data_sigma_obs=sigma_obs_np,
                    data_sigma_act=sigma_act_np,
                    data_gaze_policy=data_gaze_policy_np
                )

            val_count += 1

    return val_obs_MSE_sum / val_count, val_act_MSE_sum / val_count, val_gate_rate_sum / val_count, \
           val_gate_dims_sum / val_count, val_gate_times_sum / val_count, \
           val_obs_NLL_sum / val_count, val_act_NLL_sum / val_count, mean_variances_obs / val_count


def log_one_plot(
        data_y,
        data_target,
        data_z,
        data_actions,
        data_target_actions,
        directory_name,
        sim=-1,
        num_layers=1,
        data_g=None,
        data_sigma_obs=None,
        data_sigma_act=None,
        data_gaze_policy=None
):
    S, B, D = data_y.shape

    if sim == -1:
        sim = random.randint(0, B - 1)  # pick random sequence

    time = np.arange(0, S)

    plot_sigma_obs = data_sigma_obs is not None
    if plot_sigma_obs:
        data_sd_obs = np.sqrt(data_sigma_obs)

    plt.figure(figsize=(7, 2))

    for i in range(6):
        plt.plot(time, data_target[:, sim, i], color=colors.secondary_colors[i], linewidth=6)
        plt.plot(time, data_y[:, sim, i], color=colors.main_colors[i], linewidth=3)

        if plot_sigma_obs:
            plt.fill_between(
                time,
                data_y[:, sim, i] - data_sd_obs[:, sim, i],
                data_y[:, sim, i] + data_sd_obs[:, sim, i],
                alpha=0.15,
                facecolor=colors.main_colors[i]
            )

    if data_gaze_policy is not None:  # gaze visualization

        threes = np.ones(data_gaze_policy[:, sim, 1].shape) * 3
        eights = np.ones(data_gaze_policy[:, sim, 1].shape) * 8
        zeros = np.zeros(data_gaze_policy[:, sim, 1].shape)
        offsets = (zeros + np.where(data_gaze_policy[:, sim, 1] == 1, threes, zeros)
                   + np.where(data_gaze_policy[:, sim, 2] == 1, eights, zeros)).astype(int)

        for i in range(3):
            gaze_targets = np.zeros((S, 1))
            for t in range(S):
                gaze_targets[t, 0] = data_target[t, sim, offsets[t] + i]
            plt.plot(time, gaze_targets, color='black', linewidth=3, linestyle='dotted') # dashed lines mark gaze

    plt.ylim([0.2, 1.8])
    plt.savefig(directory_name + '_y_vs_out.png')
    plt.close()

    plot_sigma_act = data_sigma_act is not None
    if plot_sigma_act:
        data_sd_act = np.sqrt(data_sigma_act)

    time = np.arange(0, S - 1)
    plt.figure(figsize=(7, 2))
    for i in range(4):
        plt.plot(
            time,
            data_target_actions[:(S - 1), sim, i],
            color=colors.secondary_colors[i],
            linewidth=6
        )
        plt.plot(
            time,
            data_actions[:(S - 1), sim, i],
            color=colors.main_colors[i],
            linewidth=3
        )
        if plot_sigma_act:
            plt.fill_between(
                time,
                data_actions[:(S - 1), sim, i] - data_sd_act[:(S - 1), sim, i],
                data_actions[:(S - 1), sim, i] + data_sd_act[:(S - 1), sim, i],
                alpha=0.15,
                facecolor=colors.main_colors[i]
            )
    plt.savefig(directory_name + '_pred_vs_real_actions.png')
    plt.close()

    # plot all zs in one plot:
    new_z = data_z[0, :, sim, :]
    for l in range(1, num_layers):
        z_l = data_z[l, :, sim, :]
        new_z = np.concatenate((new_z, z_l), 1)

    plt.figure(figsize=(9, 2))
    plt.matshow(np.swapaxes(new_z[:, :], 0, 1), fignum=1, aspect='auto')
    plt.xlabel('t')
    plt.ylabel('dim of h')
    cbar = plt.colorbar()
    cbar.set_label('h^i_t')
    plt.savefig(directory_name + '_hs.png')
    plt.close()

    time = np.arange(0, S)
    if data_g is not None:
        plt.figure(figsize=(7, 2))
        plt.plot(time, np.clip(np.sum(data_g[:, sim, :], 1), 0, 1), color='k', linewidth=6)
        plt.ylim([-0.1, 1.1])
        plt.savefig(directory_name + '_gates.png')
        plt.close()


