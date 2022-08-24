import os

def create_res_dirs(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    checkpoint_dir = target_dir + "/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    plot_dir = target_dir + "/plots/"
    os.makedirs(plot_dir, exist_ok=True)
    metrics_dir = target_dir + "/metrics/"
    os.makedirs(metrics_dir, exist_ok=True)
    return checkpoint_dir, plot_dir, metrics_dir