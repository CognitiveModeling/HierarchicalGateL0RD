import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import models.forward_inverse_model as FIM
import models.skip_net as SkipNet

from models.skip_data_gen import sequence_to_skip_data_gradfree, batch_to_skip_inputs
from dataloaders.fetch_event_dataset import create_event_dataloaders, create_apex_dataloaders
from utils.set_rs import set_rs
from utils.gaze_modeling_utils import add_gaze_based_noise_alternating
from utils.torch_utils import mean_euclidean_distance
from utils.os_managment import create_res_dirs


def main_skip_training(params):

    print("Starting FPP skip network training with params ", params)

    rand_seed = params.rs
    set_rs(rand_seed)

    # create directories for logging
    fim_model_dir = params.model_dir  + '/' + str(rand_seed) + '/checkpoints/'
    target_dir = params.skip_model_dir  + '/' + str(rand_seed) + '/'
    checkpoint_dir, plot_dir, metrics_dir = create_res_dirs(target_dir)

    # Dimensionality of the scenario
    action_dim = 4
    state_dim = 11

    # factors for scaling predictions
    output_factor = 0.1  # observation prediction
    output_factor2 = 0.1  # action prediction

    # Gauss network?
    gaussian_outputs = params.gauss_net

    # gaze related
    add_gaze = False
    gaze_dim = 0
    if params.add_gaze:
        add_gaze = True
        gaze_dim = 3
    gaze_noise_sd = params.gaze_noise_sd
    focus_noise_sd = params.focus_noise_sd
    gaze_alternations = params.alt_gaze_num
    start_hand = params.start_hand

    # create the low-level network
    gate_net = FIM.ForwardInverseModel(
        reg_lambda=params.reg_lambda,
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

    # Skip Network
    skip_input_dim = state_dim + params.latent_dim + gaze_dim
    skip_output_dim = state_dim
    skip_net = SkipNet.SkipNet(
        input_dim=skip_input_dim,
        output_dim=skip_output_dim,
        feature_dim=params.skip_feature_dim,
        num_layers=params.skip_num_layers,
        gauss_net=params.skip_gauss_net,
        nll_beta=params.skip_nll_beta
    )
    skip_optimizer = skip_net.get_optimizer(params.skip_lr)
    skip_predict_deltas = params.skip_predict_deltas
    skip_ignore_single_steps = params.skip_ignore_single_steps

    # Data related
    dataset_split_rs = params.dataset_split_rs
    num_data_per_dataset_train = 3200
    num_data_per_dataset_test = 3200
    b_size_per_dataset = 64
    seq_len = params.seq_len

    if params.apex_data:
        # triple every number
        num_data_per_dataset_train = int(num_data_per_dataset_train * 3)
        num_data_per_dataset_test = int(num_data_per_dataset_test)
        train_batch_size_per_dataset = int(b_size_per_dataset * 3)
        val_batch_size_per_dataset = int(b_size_per_dataset * 5)
        train_dataloaders, val_dataloaders, test_dataloaders = create_apex_dataloaders(
            seq_len=seq_len,
            dataset_split_rs=dataset_split_rs,
            num_data_train=num_data_per_dataset_train,
            num_data_test=num_data_per_dataset_test,
            val_batch_size=val_batch_size_per_dataset,
            train_batch_size=train_batch_size_per_dataset
        )
    else:
        train_batch_size_per_dataset = b_size_per_dataset
        val_batch_size_per_dataset = b_size_per_dataset
        train_dataloaders, val_dataloaders, test_dataloaders = create_event_dataloaders(
            seq_len=seq_len,
            dataset_split_rs=dataset_split_rs,
            num_data_train=num_data_per_dataset_train,
            num_data_test=num_data_per_dataset_test,
            train_batch_size=train_batch_size_per_dataset,
            test_batch_size=val_batch_size_per_dataset
        )

    num_train_datasets = len(train_dataloaders)
    num_datasets = len(val_dataloaders)
    overall_batch_size = int(val_batch_size_per_dataset * num_datasets)
    overall_train_batch_size = int(train_batch_size_per_dataset * num_train_datasets)

    t_validation = 100
    t_plots = 10
    t_save_metrics = 1
    num_validations = 10 * params.train_len
    num_epochs = t_validation * num_validations

    # All metrics

    val_skip_MSE_over_t = np.zeros(num_validations, dtype='float64')
    val_skip_NLL_over_t = np.zeros(num_validations, dtype='float64')

    val_skip_distance_over_t = np.zeros((len(val_dataloaders), 3, num_validations), dtype='float64')
    val_skip_variance_over_t = np.zeros((len(val_dataloaders), 3, num_validations), dtype='float64')

    test_skip_MSE_over_t = np.zeros(num_validations, dtype='float64')
    test_skip_NLL_over_t = np.zeros(num_validations, dtype='float64')

    test_skip_distance_over_t = np.zeros((len(test_dataloaders), 3, num_validations), dtype='float64')
    test_skip_variance_over_t = np.zeros((len(test_dataloaders), 3, num_validations), dtype='float64')

    loss_skip_net = np.zeros(num_epochs, dtype='float64')

    # Which indices are compared when evaluating skip predictions?
    relevant_pos_skip = [[0, 1, 2], [0, 1, 2], [0, 1, 2]] # We compare skip predicted hand positions to current...
    relevant_pos_input = [[0, 1, 2], [3, 4, 5], [8, 9, 10]] # ... positions of (1.) hand, (2.) object & (3.) goal


    epoch_start = 0
    validations = 0

    fname = os.path.join(checkpoint_dir, "checkpoint")

    if os.path.isfile(fname):
        # We load the model
        dir_name_checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(dir_name_checkpoint)
        skip_net.load_state_dict(checkpoint['skip_net_state_dict'])
        skip_optimizer.load_state_dict(checkpoint['skip_optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        validations = checkpoint['validations']

        # load skip metrics:
        val_skip_MSE_np_file = os.path.join(metrics_dir, "val_skip_MSE_np.npy")
        val_skip_MSE_over_t[:validations] = np.load(val_skip_MSE_np_file)[:validations]

        val_skip_NLL_np_file = os.path.join(metrics_dir, "val_skip_NLL_np.npy")
        val_skip_NLL_over_t[:validations] = np.load(val_skip_NLL_np_file)[:validations]


        val_skip_distances_np_file = os.path.join(metrics_dir, "val_skip_distance_np.npy")
        val_skip_distance_over_t[:, :, :validations] = np.load(val_skip_distances_np_file)[:, :, :validations]

        val_skip_variance_np_file = os.path.join(metrics_dir, "val_skip_variance_np.npy")
        val_skip_variance_over_t[:, :, :validations] = np.load(val_skip_variance_np_file)[:, :, :validations]

        test_skip_MSE_np_file = os.path.join(metrics_dir, "test_skip_MSE_np.npy")
        test_skip_MSE_over_t[:validations] = np.load(test_skip_MSE_np_file)[:validations]

        test_skip_NLL_np_file = os.path.join(metrics_dir, "test_skip_NLL_np.npy")
        test_skip_NLL_over_t[:validations] = np.load(test_skip_NLL_np_file)[:validations]

        test_skip_distances_np_file = os.path.join(metrics_dir, "test_skip_distance_np.npy")
        test_skip_distance_over_t[:, :, :validations] = np.load(test_skip_distances_np_file)[:, :, :validations]

        test_skip_variance_np_file = os.path.join(metrics_dir, "test_skip_variance_np.npy")
        test_skip_variance_over_t[:, :, :validations] = np.load(test_skip_variance_np_file)[:, :, :validations]

        loss_skip_net_file = os.path.join(metrics_dir, "loss_skip_net_np.npy")
        loss_skip_net[:epoch_start] = np.load(loss_skip_net_file)[:epoch_start]

    epoch = epoch_start
    print("------------ ")
    for epoch in range(epoch_start, num_epochs):


        if epoch%t_validation == 0:

            # Load FIM-GateL0RD from the same evaluation
            dir_name_gatel0rd_checkpoint = os.path.join(fim_model_dir, "checkpoint_v" + str(validations+1))
            if not os.path.isfile(dir_name_gatel0rd_checkpoint):
                assert False, "Model caught up with GateL0RD training: " + dir_name_gatel0rd_checkpoint
            checkpoint = torch.load(dir_name_gatel0rd_checkpoint)
            gate_net.load_state_dict(checkpoint['gate_net_state_dict'])

            print("Evaluation ", validations, " after epoch ", epoch)

            # VALIDATION
            # Skip network validation
            # Error measures (MSE, NLL) to validation targets
            val_skip_MSE_over_t[validations], val_skip_NLL_over_t[validations] = eval_skip_fc(
                dataloaders=val_dataloaders,
                network=gate_net,
                skip_network=skip_net,
                seq_len=seq_len,
                batch_size=overall_batch_size,
                start_steps=1,
                n_step_prediction=False,
                mode='observation',
                factor_output=output_factor,
                skip_predict_deltas=skip_predict_deltas,
                skip_output_dim=skip_output_dim,
                skip_example_plot=(validations % t_plots == 0),
                skip_example_plot_directory_name=plot_dir + '/plot_val_' + str(validations) + '_skip_',
                factor_output2=output_factor2,
                add_gaze=add_gaze,
                gaze_noise_sd=gaze_noise_sd,
                focus_noise_sd=focus_noise_sd,
                num_alternations=gaze_alternations,
                start_hand=start_hand,
                skip_ignore_single_steps=skip_ignore_single_steps,
            )

            # Per dataset compare skip predictions of hand to current positions of all entities for t = 1
            for val_dl_i,val_dl in enumerate(val_dataloaders):
                val_skip_distance_i, val_skip_var_i = compare_skip_to_positions(
                    dataloaders=[val_dl],
                    network=gate_net,
                    skip_network=skip_net,
                    skip_t=1,
                    seq_len=seq_len,
                    batch_size=val_batch_size_per_dataset,
                    start_steps=1,
                    n_step_prediction=False,
                    mode='observation',
                    factor_output=output_factor,
                    skip_predict_deltas=skip_predict_deltas,
                    skip_output_dim=skip_output_dim,
                    relevant_skip_dims=relevant_pos_skip,
                    relevant_target_dims=relevant_pos_input,
                    factor_output2=output_factor2,
                    add_gaze=add_gaze,
                    gaze_noise_sd=gaze_noise_sd,
                    num_alternations=gaze_alternations,
                    start_hand=start_hand,
                    focus_noise_sd=focus_noise_sd,
                )
                val_skip_distance_over_t[val_dl_i, :, validations] = val_skip_distance_i
                val_skip_variance_over_t[val_dl_i, :, validations] = val_skip_var_i

            # TESTING
            # Skip network validation
            test_skip_MSE_over_t[validations], test_skip_NLL_over_t[validations] = eval_skip_fc(
                dataloaders=test_dataloaders,
                network=gate_net,
                skip_network=skip_net,
                seq_len=seq_len,
                batch_size=overall_batch_size,
                start_steps=1,
                n_step_prediction=False,
                mode='observation',
                factor_output=output_factor,
                skip_predict_deltas=skip_predict_deltas,
                skip_output_dim=skip_output_dim,
                skip_example_plot=False,
                skip_example_plot_directory_name='',
                factor_output2=output_factor2,
                add_gaze=add_gaze,
                gaze_noise_sd=gaze_noise_sd,
                num_alternations=gaze_alternations,
                start_hand=start_hand,
                focus_noise_sd=focus_noise_sd,
                skip_ignore_single_steps=skip_ignore_single_steps,
            )

            for test_dl_i, test_dl in enumerate(test_dataloaders):
                test_skip_distance_i, test_skip_var_i = compare_skip_to_positions(
                    dataloaders=[test_dl],
                    network=gate_net,
                    skip_network=skip_net,
                    skip_t=1,
                    seq_len=seq_len,
                    batch_size=val_batch_size_per_dataset,
                    start_steps=1,
                    n_step_prediction=False,
                    mode='observation',
                    factor_output=output_factor,
                    skip_predict_deltas=skip_predict_deltas,
                    skip_output_dim=skip_output_dim,
                    relevant_skip_dims=relevant_pos_skip,
                    relevant_target_dims=relevant_pos_input,
                    factor_output2=output_factor2,
                    add_gaze=add_gaze,
                    gaze_noise_sd=gaze_noise_sd,
                    num_alternations=gaze_alternations,
                    start_hand=start_hand,
                    focus_noise_sd=focus_noise_sd,
                )
                test_skip_distance_over_t[test_dl_i, :, validations] = test_skip_distance_i
                test_skip_variance_over_t[test_dl_i, :, validations] = test_skip_var_i

            if validations % t_save_metrics == 0:
                print("Saving metrics ...")
                # save all metrics:
                val_skip_MSE_np_file = os.path.join(metrics_dir, "val_skip_MSE_np.npy")
                np.save(val_skip_MSE_np_file, val_skip_MSE_over_t)

                val_skip_NLL_np_file = os.path.join(metrics_dir, "val_skip_NLL_np.npy")
                np.save(val_skip_NLL_np_file, val_skip_NLL_over_t)

                val_skip_distance_np_file = os.path.join(metrics_dir, "val_skip_distance_np.npy")
                np.save(val_skip_distance_np_file, val_skip_distance_over_t)

                val_skip_variance_np_file = os.path.join(metrics_dir, "val_skip_variance_np.npy")
                np.save(val_skip_variance_np_file, val_skip_variance_over_t)

                test_skip_MSE_np_file = os.path.join(metrics_dir, "test_skip_MSE_np.npy")
                np.save(test_skip_MSE_np_file, test_skip_MSE_over_t)

                test_skip_NLL_np_file = os.path.join(metrics_dir, "test_skip_NLL_np.npy")
                np.save(test_skip_NLL_np_file, test_skip_NLL_over_t)

                test_skip_distance_np_file = os.path.join(metrics_dir, "test_skip_distance_np.npy")
                np.save(test_skip_distance_np_file, test_skip_distance_over_t)

                test_skip_variance_np_file = os.path.join(metrics_dir, "test_skip_variance_np.npy")
                np.save(test_skip_variance_np_file, test_skip_variance_over_t)

                loss_skip_net_file = os.path.join(metrics_dir, "loss_skip_net_np.npy")
                np.save(loss_skip_net_file, loss_skip_net)

            validations += 1
            # checkpointing
            for checkpoint_name in ["checkpoint", "checkpoint_v" + str(validations)]:
                dir_name_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
                # save a checkpoint
                torch.save({
                    'epoch': epoch,
                    'skip_net_state_dict': skip_net.state_dict(),
                    'skip_optimizer_state_dict': skip_optimizer.state_dict(),
                    'validations': validations
                }, dir_name_checkpoint)

        if validations < num_validations:

            # load GateL0RD from last evaluation
            dir_name_gatel0rd_checkpoint = os.path.join(fim_model_dir, "checkpoint_v" + str(validations))
            if not os.path.isfile(dir_name_gatel0rd_checkpoint):
                assert False, "Model caught up with GateL0RD training: " + dir_name_gatel0rd_checkpoint
            checkpoint = torch.load(dir_name_gatel0rd_checkpoint)
            gate_net.load_state_dict(checkpoint['gate_net_state_dict'])

            # train Skip Net:
            gate_net = gate_net.eval()
            skip_net = skip_net.train()
            skip_ss = np.ones((seq_len, overall_train_batch_size))

            loss_skip_net_sum = 0.0
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
                next_obs = obs + torch.cat(list_delta_obs, dim=1).float()
                clear_obs = obs.clone()
                s, b, _ = obs.shape
                skip_net.zero_grad()
                skip_optimizer.zero_grad()

                # create skip training data from FIM
                with torch.no_grad():
                    gaze_policy = None
                    if add_gaze:
                        gaze_policy, obs, next_obs = add_gaze_based_noise_alternating(
                            obs, next_obs,
                            gaze_noise_sd,
                            dim=-1,
                            num_alternations=gaze_alternations,
                            start_hand=start_hand,
                            focus_noise_sd=focus_noise_sd
                        )

                    y_obs, z, gate_reg, _, _, y_act, _, _ = gate_net.forward_n_step(
                        obs_batch=obs,
                        train_schedule=skip_ss,
                        predict_o_deltas=True,
                        factor_o_delta=output_factor,
                        action_batch=actions,
                        mode='observation',
                        predict_a_deltas=False,
                        factor_a_delta=output_factor2,
                        additional_info=gaze_policy
                    )

                    act_input = None
                    if add_gaze:
                        act_input = gaze_policy

                    skip_inps, skip_tars = sequence_to_skip_data_gradfree(
                        xs=obs,
                        gs=gate_reg[0, :, :, :],
                        hs=z[0, :, :, :],
                        add_inps=act_input,
                        ignore_single_steps=skip_ignore_single_steps
                    )
                    if add_gaze: # clear skip targets
                        _, skip_tars = sequence_to_skip_data_gradfree(
                            xs=clear_obs,
                            gs=gate_reg[0, :, :, :],
                            hs=z[0, :, :, :],
                            add_inps=act_input,
                            ignore_single_steps=skip_ignore_single_steps
                        )

                if len(skip_inps) > 0: # there exists skip training data
                    skip_ys, skip_sigmas = skip_net.forward(skip_inps)
                    skip_outs = skip_ys
                    if skip_predict_deltas:
                        skip_outs[:, :skip_output_dim] += skip_inps[:, :skip_output_dim]

                    skip_loss = skip_net.loss(out=skip_outs, tar=skip_tars, sigmas=skip_sigmas)
                    skip_loss.backward()
                    skip_optimizer.step()
                    train_count += 1
                    loss_skip_net_sum += skip_loss.detach().item()

            if train_count > 0:
                loss_skip_net[epoch] = loss_skip_net_sum/train_count
            else:
                loss_skip_net[epoch] = 9999999 # fill with high loss if not trained

            print("Training epoch ", epoch, "/", num_epochs, " done, mean loss = ", loss_skip_net_sum/train_count)


    # save everything
    final_checkpoint_names = ["checkpoint",  "final_checkpoint"]
    for fname in final_checkpoint_names:
        dir_name_checkpoint = os.path.join(checkpoint_dir, fname)

        # save a checkpoint
        torch.save({
            'epoch': epoch + 1,
            'skip_net_state_dict': skip_net.state_dict(),
            'skip_optimizer_state_dict': skip_optimizer.state_dict(),
            'validations': validations
        }, dir_name_checkpoint)

    # Save all metrics:
    val_skip_MSE_np_file = os.path.join(metrics_dir, "val_skip_MSE_np.npy")
    np.save(val_skip_MSE_np_file, val_skip_MSE_over_t)

    val_skip_NLL_np_file = os.path.join(metrics_dir, "val_skip_NLL_np.npy")
    np.save(val_skip_NLL_np_file, val_skip_NLL_over_t)

    val_skip_distance_np_file = os.path.join(metrics_dir, "val_skip_distance_np.npy")
    np.save(val_skip_distance_np_file, val_skip_distance_over_t)

    val_skip_variance_np_file = os.path.join(metrics_dir, "val_skip_variance_np.npy")
    np.save(val_skip_variance_np_file, val_skip_variance_over_t)

    test_skip_MSE_np_file = os.path.join(metrics_dir, "test_skip_MSE_np.npy")
    np.save(test_skip_MSE_np_file, test_skip_MSE_over_t)

    test_skip_NLL_np_file = os.path.join(metrics_dir, "test_skip_NLL_np.npy")
    np.save(test_skip_NLL_np_file, test_skip_NLL_over_t)

    test_skip_distance_np_file = os.path.join(metrics_dir, "test_skip_distance_np.npy")
    np.save(test_skip_distance_np_file, test_skip_distance_over_t)

    test_skip_variance_np_file = os.path.join(metrics_dir, "test_skip_variance_np.npy")
    np.save(test_skip_variance_np_file, test_skip_variance_over_t)

    loss_skip_net_file = os.path.join(metrics_dir, "loss_skip_net_np.npy")
    np.save(loss_skip_net_file, loss_skip_net)



def eval_skip_fc(
        dataloaders,
        network,
        skip_network,
        seq_len,
        batch_size,
        start_steps,
        n_step_prediction,
        mode,
        factor_output,
        factor_output2,
        skip_predict_deltas,
        skip_output_dim,
        skip_example_plot=False,
        skip_example_plot_directory_name='',
        add_gaze=False,
        focus_noise_sd=0.0,
        gaze_noise_sd=-1,
        gaze_dim=-1,
        num_alternations=3,
        start_hand=False,
        skip_ignore_single_steps=True
):

    network = network.eval()
    skip_network = skip_network.eval()

    # 1step vs N step prediction?
    val_ss = np.ones((seq_len, batch_size))
    if n_step_prediction:
        val_ss[start_steps:seq_len, :] = -1

    val_count = 0
    plot_count = 0
    val_skip_MSE_sum = 0.0
    val_skip_NLL = 0.0

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
            next_obs = obs + torch.cat(list_delta_obs, dim=1).float()
            clear_obs = obs.clone()
            gaze_policy = None
            if add_gaze:

                gaze_policy, obs, next_obs = add_gaze_based_noise_alternating(
                    obs,
                    next_obs,
                    gaze_noise_sd,
                    dim=gaze_dim,
                    focus_noise_sd=focus_noise_sd,
                    num_alternations=num_alternations,
                    start_hand=start_hand
                )

            s, b, _ = obs.shape
            _, z, gate_reg, _, _, _, _, _ = network.forward_n_step(
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
            act_input = None
            if add_gaze:
                act_input = gaze_policy

            if skip_example_plot and plot_count==0:

                if act_input is not None:
                    plot_act_input = act_input[:, 0:1, :]
                else:
                    plot_act_input = None

                # One sequence is plotted
                skip_inps, skip_tars = sequence_to_skip_data_gradfree(
                    xs=obs[:, 0:1, :],
                    gs=gate_reg[0, :, 0:1, :],
                    hs=z[0, :, 0:1, :],
                    add_inps=plot_act_input,
                    ignore_single_steps=skip_ignore_single_steps
                )

                if add_gaze: # clear skip targets
                    _, skip_tars = sequence_to_skip_data_gradfree(
                        xs=clear_obs[:, 0:1, :],
                        gs=gate_reg[0, :, 0:1, :],
                        hs=z[0, :, 0:1, :],
                        add_inps=plot_act_input,
                        ignore_single_steps=skip_ignore_single_steps
                    )

                if len(skip_inps) > 0:
                    skip_ys, skip_sigmas = skip_network.forward(skip_inps)
                    skip_outs = skip_ys
                    if skip_predict_deltas:
                        skip_outs[:, :skip_output_dim] += skip_inps[:, :skip_output_dim]
                    log_trajectory_skip(
                        real_x=obs[:, 0:1, :].detach().numpy(),
                        tar_skip_x=skip_tars.detach().numpy(),
                        pred_skip_x=skip_outs.detach().numpy(),
                        seq_len=seq_len,
                        directory_name=skip_example_plot_directory_name
                    )
                    plot_count += 1

            skip_inps, skip_tars = sequence_to_skip_data_gradfree(
                xs=obs,
                gs=gate_reg[0, :, :, :],
                hs=z[0, :, :, :],
                add_inps=act_input,
                ignore_single_steps=skip_ignore_single_steps
            )
            if add_gaze: # clear skip targets
                _, skip_tars = sequence_to_skip_data_gradfree(
                    xs=clear_obs,
                    gs=gate_reg[0, :, :, :],
                    hs=z[0, :, :, :],
                    add_inps=act_input,
                    ignore_single_steps=skip_ignore_single_steps
                )

            if len(skip_inps) > 0:
                skip_ys, skip_sigmas = skip_network.forward(skip_inps)
                skip_outs = skip_ys
                if skip_predict_deltas:
                    skip_outs[:, :skip_output_dim] += skip_inps[:, :skip_output_dim]
                val_skip_MSE_sum += skip_network.MSE(skip_tars, skip_outs).detach().item()
                if skip_sigmas is not None:
                    # NLL analysis
                    val_skip_NLL += skip_network.NLL(skip_ys, skip_sigmas, skip_tars, ignore_beta=True).detach().item()
                val_count += 1

    if val_count == 0:
        return 100, 100  # DEFAULT HIGH PREDICTION ERROR

    return val_skip_MSE_sum/val_count, val_skip_NLL/val_count


def compare_skip_to_positions(
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
        relevant_skip_dims,
        relevant_target_dims,
        add_gaze=False,
        focus_noise_sd=0.0,
        gaze_noise_sd=-1,
        gaze_dim=-1,
        num_alternations=3,
        start_hand=False,
):

    num_relevant_dims = len(relevant_target_dims)
    network = network.eval()
    skip_network = skip_network.eval()

    # 1step vs N step prediction?
    val_ss = np.ones((skip_t, batch_size))
    if n_step_prediction:
        val_ss[start_steps:skip_t, :] = -1

    val_count = 0
    val_skip_distance = np.zeros(num_relevant_dims)
    val_skip_variances = np.zeros(num_relevant_dims)

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
            next_obs = obs + torch.cat(list_delta_obs, dim=1).float()

            gaze_policy = None
            if add_gaze:
                gaze_policy, obs, next_obs = add_gaze_based_noise_alternating(
                    obs,
                    next_obs,
                    gaze_noise_sd,
                    dim=gaze_dim,
                    focus_noise_sd=focus_noise_sd,
                    num_alternations=num_alternations,
                    start_hand=start_hand
                )

            s, b, _ = obs.shape

            add_info = gaze_policy
            if add_gaze:
                add_info = gaze_policy[:skip_t, :, :]

            _, z, gate_reg, _, _, _, _, _ = network.forward_n_step(
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

            act_input = None
            if add_gaze:
                act_input = gaze_policy

            if act_input is not None:
                batch_act_input = act_input[skip_t-1, :, :]
            else:
                batch_act_input = None

            skip_inps = batch_to_skip_inputs(
                xs=obs[skip_t-1, :, :],
                hs=z[0, skip_t-1, :, :],
                add_inps=batch_act_input
            )
            skip_ys, skip_sigmas = skip_network.forward(skip_inps)
            skip_outs = skip_ys
            if skip_predict_deltas:
                skip_outs[:, :skip_output_dim] += skip_inps[:, :skip_output_dim]

            for n in range(num_relevant_dims):
                val_skip_distance[n] += mean_euclidean_distance(
                    obs[skip_t-1, :, relevant_target_dims[n]],
                    skip_outs[:, relevant_skip_dims[n]]
                ).detach().item()

                if skip_sigmas is not None:
                    val_skip_variances[n] += torch.mean(skip_sigmas[:, relevant_target_dims[n]])
            val_count += 1
    return val_skip_distance/val_count, val_skip_variances/val_count


def log_trajectory_skip(
        real_x,
        tar_skip_x,
        pred_skip_x,
        seq_len,
        directory_name
):

    assert len(real_x.shape) == 3 and real_x.shape[1] == 1

    traj1_np = real_x[0:seq_len, 0, 0:6]
    traj2_np = tar_skip_x[:, 0:6]
    traj3_np = pred_skip_x[:, 0:6]

    max_x = max([np.max(traj1_np[:, 0]), np.max(traj2_np[:, 0]), np.max(traj3_np[:, 0]),
                 np.max(traj1_np[:, 3]), np.max(traj2_np[:, 3]), np.max(traj3_np[:, 3])]) + 0.01
    max_y = max([np.max(traj1_np[:, 1]), np.max(traj2_np[:, 1]), np.max(traj3_np[:, 1]),
                 np.max(traj1_np[:, 4]), np.max(traj2_np[:, 4]), np.max(traj3_np[:, 4])]) + 0.01
    max_z = max([np.max(traj1_np[:, 2]), np.max(traj2_np[:, 2]), np.max(traj3_np[:, 2]),
                 np.max(traj1_np[:, 5]), np.max(traj2_np[:, 5]), np.max(traj3_np[:, 5])]) + 0.01

    min_x = min([np.min(traj1_np[:, 0]), np.min(traj2_np[:, 0]), np.min(traj3_np[:, 0]),
                 np.min(traj1_np[:, 3]), np.min(traj2_np[:, 3]), np.min(traj3_np[:, 3])]) - 0.01
    min_y = min([np.min(traj1_np[:, 1]), np.min(traj2_np[:, 1]), np.min(traj3_np[:, 1]),
                 np.min(traj1_np[:, 4]), np.min(traj2_np[:, 4]), np.min(traj3_np[:, 4])]) - 0.01
    min_z = min([np.min(traj1_np[:, 2]), np.min(traj2_np[:, 2]), np.min(traj3_np[:, 2]),
                 np.min(traj1_np[:, 5]), np.min(traj2_np[:, 5]), np.min(traj3_np[:, 5])]) - 0.01

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(12, 6)

    c = np.arange(0, seq_len)
    cm2 = plt.cm.get_cmap('autumn')
    cm = plt.cm.get_cmap('winter')

    sc = ax.scatter(traj1_np[:, 0], traj1_np[:, 1], traj1_np[:, 2], s=100, c=c, cmap=cm)
    sc2 = ax.scatter(traj1_np[:, 3], traj1_np[:, 4], traj1_np[:, 5], s=100, c=c, cmap=cm2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Real traj')
    ax.set_zlabel('z')
    cbar = plt.colorbar(sc)
    cbar.set_label('gripper')
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('object')
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])
    plt.savefig(directory_name + '_pred_traj.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(12, 6)
    c = np.arange(0, traj2_np.shape[0])

    cm = plt.cm.get_cmap('summer')
    cm2 = plt.cm.get_cmap('copper')

    sc = ax.scatter(traj2_np[:, 0], traj2_np[:, 1], traj2_np[:, 2], s=100, c=c, cmap=cm)
    sc2 = ax.scatter(traj2_np[:, 3], traj2_np[:, 4], traj2_np[:, 5], s=100, c=c, cmap=cm2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Target skip traj')
    ax.set_zlabel('z')
    cbar = plt.colorbar(sc)
    cbar.set_label('gripper')
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('object')
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])
    plt.savefig(directory_name + '_skip_tar_traj.png')
    plt.close()

    traj2_np = pred_skip_x[:, 0:6]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(12, 6)
    c = np.arange(0, traj2_np.shape[0])
    cm2 = plt.cm.get_cmap('Wistia')
    cm = plt.cm.get_cmap('cool')

    sc = ax.scatter(traj2_np[:, 0], traj2_np[:, 1], traj2_np[:, 2], s=100, c=c, cmap=cm)
    sc2 = ax.scatter(traj2_np[:, 3], traj2_np[:, 4], traj2_np[:, 5], s=100, c=c, cmap=cm2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pred. skip traj')
    ax.set_zlabel('z')
    cbar = plt.colorbar(sc)
    cbar.set_label('gripper')
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('object')
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])
    plt.savefig(directory_name + '_skip_pred_traj.png')
    plt.close()
