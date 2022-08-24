import torch
import numpy as np
from torch.utils.data import Dataset
import utils.rolloutbuffer as rolloutbuffer


class FPPDataset(Dataset):
    """
    Dataset for all kinds of data from Fetch Pick&Place simulation
    """

    def __init__(self, relevant_dims, seq_len, path=None, name=None, data=None):
        super(FPPDataset, self).__init__()

        self.seq_len = seq_len
        if name is not None and path is not None:
            action_data = np.load(path + name + 'actions.npy')
            observation_data = np.load(path + name + 'obs.npy')
            next_observation_data = np.load(path + name + 'next_obs.npy')
        else:
            assert data is not None
            observation_data, action_data, next_observation_data = data

        act_shape = action_data.shape
        assert act_shape[1] == 50
        assert act_shape[2] == 4
        self.num_data = observation_data.shape[0]
        assert act_shape[0] == self.num_data

        self.act_data = action_data[:, 0:self.seq_len, :]
        self.obs_data = observation_data[:, 0:self.seq_len, relevant_dims]
        self.next_obs_data = next_observation_data[:, 0:self.seq_len, relevant_dims]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.obs_data[idx, :, :], self.act_data[idx, :, :], (self.next_obs_data[idx, :, :] - self.obs_data[idx, :, :])


def create_event_dataloaders(seq_len, dataset_split_rs, num_data_train, num_data_test, train_batch_size, test_batch_size,):
    # Creates dataloaders for event sequence data

    # observation dimensions used are hand pos (0-2), object pos (3-5), gripper opening (9-10), goal pos (28-30)
    relevant_dims = np.array([0, 1, 2, 3, 4, 5, 9, 10, 28, 29, 30])

    path = 'data/scripted/varying_table/'  # TODO adjust path
    dataset_names = ['grasp', 'point', 'stretch']

    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    for name in dataset_names:
        dataset = FPPDataset(path=path, name=name+'/', relevant_dims=relevant_dims, seq_len=seq_len)
        num_data_ignore = len(dataset) - 2 * num_data_test - num_data_train
        train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(
            dataset,
            [num_data_train, num_data_test, num_data_test, num_data_ignore],
            generator=torch.Generator().manual_seed(dataset_split_rs)
        )

        train_dataloaders.append(torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True))
        val_dataloaders.append(torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False))
        test_dataloaders.append(torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False))

    return train_dataloaders, val_dataloaders, test_dataloaders

def create_apex_dataloaders(seq_len, dataset_split_rs, num_data_train, num_data_test, val_batch_size, train_batch_size):
    # Creates dataloaders for apex data

    # observation dimensions used are hand pos (0-2), object pos (3-5), gripper opening (9-10), goal pos (28-30)
    relevant_dims = np.array([0, 1, 2, 3, 4, 5, 9, 10, 28, 29, 30])
    num_apex_datasets = 1

    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []

    # APEX filtered for clean grasp sequences for visualization
    path = 'data/APEX/' # TODO adjust path
    name = 'grasp/'
    dataset_grasp = FPPDataset(path=path, name=name, relevant_dims=relevant_dims, seq_len=seq_len)
    num_data_ignore_grasp = len(dataset_grasp) - 2 * num_data_test
    val_dataset_grasp, test_dataset_grasp, _ = torch.utils.data.random_split(
        dataset_grasp,
        [num_data_test, num_data_test,
         num_data_ignore_grasp],
        generator=torch.Generator().manual_seed(dataset_split_rs)
    )
    val_dataloaders.append(torch.utils.data.DataLoader(val_dataset_grasp, batch_size=val_batch_size, shuffle=False))
    test_dataloaders.append(torch.utils.data.DataLoader(test_dataset_grasp, batch_size=val_batch_size, shuffle=False))


    # full APEX data used for training and numerical evaluation
    dataset = FPPDataset(relevant_dims=relevant_dims, seq_len=seq_len, data=load_apex_data(num_apex_datasets))
    num_data_ignore = len(dataset) - 2 * num_data_test - num_data_train
    train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(
        dataset,
        [num_data_train, num_data_test, num_data_test, num_data_ignore],
        generator=torch.Generator().manual_seed(dataset_split_rs)
    )
    train_dataloaders.append(torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True))
    val_dataloaders.append(torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False))
    test_dataloaders.append(torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False))
    return train_dataloaders, val_dataloaders, test_dataloaders


def load_apex_data(num_apex_datasets):
    data_path = "data/APEX/full/"  # TODO adjust path
    obs_list = []
    next_obs_list = []
    acts_list = []
    env_list = []
    num_datasets = num_apex_datasets
    for i in range(num_datasets):
        f_path = data_path + str(i + 1)
        with open(f_path, "rb") as f:
            r = rolloutbuffer.rollout_load(f)
            obs_list.append(r.as_array("observations"))
            next_obs_list.append(r.as_array("next_observations"))
            acts_list.append(r.as_array("actions"))
            env_list.append(r.as_array('env_states'))
    all_obs = obs_list[0]
    all_next_obs = next_obs_list[0]
    all_acts = acts_list[0]
    all_envs = env_list[0]
    print("Loading apex data # 0")
    for i in range(num_datasets - 1):
        print("Loading apex data #", i + 1)
        all_obs = np.concatenate((all_obs, obs_list[i + 1]), 0)
        all_next_obs = np.concatenate((all_next_obs, next_obs_list[i + 1]), 0)
        all_acts = np.concatenate((all_acts, acts_list[i + 1]), 0)
        all_envs = np.concatenate((all_envs, env_list[i + 1]), 0)

    all_obs_with_goals = np.concatenate((all_obs, all_envs[:, :, -3:]), 2)
    all_next_obs_with_goals = np.concatenate((all_next_obs, all_envs[:, :, -3:]), 2)

    return all_obs_with_goals, all_acts, all_next_obs_with_goals





