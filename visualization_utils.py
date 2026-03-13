import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, List
import umap
import seaborn as sns

def plot_combined_data(data_points, generated_points, output_file="plot.png", highlight_times=[0., 0.5, 1.]):

    cmap = plt.cm.viridis
    sns.set_theme(style="white")

    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, 10))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
        'font.size': 12,
        'axes.titlesize': 10,
        'axes.labelsize': 12
    })

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    T_data = data_points.shape[0]
    T_gen, batch_size, dim = generated_points.shape
    T_high = len(highlight_times)

    colors = cmap(np.linspace(0, 1, T_high))

    # marker styling (same trick as original code)
    size = 50
    size_x = 300
    outline_width = (0.3, 0.1)
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2

    # --------------------
    # Ground truth points
    # --------------------
    for i in range(T_data):
        x = data_points[i, :, 0]
        y = data_points[i, :, 1]

        ax.scatter(
            x, y,
            marker='X',
            s=size_x,
            color=colors[i],
            alpha=0.2,
            linewidths=0
        )

    # --------------------
    # Generated points
    # --------------------
    
    # Highlight specific time points if requested
    generated = np.zeros([T_high, batch_size, dim])
    t_samples_np = np.linspace(highlight_times[0], highlight_times[-1], T_gen) 
    for idx_h, t_high in enumerate(highlight_times):
        closest_idx = np.argmin(np.abs(t_samples_np - t_high))
        generated[idx_h] = generated_points[closest_idx]

    mass = generated[:, :, 2]
    mass_max_value = np.percentile(mass, 99)
    norm_mass = mass / (mass_max_value + 1e-8)
    norm_mass[norm_mass > 1.] = 1.
    minimum_value = 0.
    rescaled_minimum = 0.75
    mask = norm_mass >= minimum_value
    # rescale values >= 0.3 to [0.8, 1]
    norm_mass[mask] = rescaled_minimum + (norm_mass[mask] - minimum_value) / (1 - minimum_value) * (1 - rescaled_minimum)
    # set the rest to 0
    norm_mass[~mask] = 0
    # norm_mass[norm_mass < 0.3] = 0.

    for i in range(T_high):

        x = generated[i, :, 0]
        y = generated[i, :, 1]
        alpha = norm_mass[i]

        # outer black border
        ax.scatter(
            x, y,
            s=bg_size,
            c='black',
            alpha=alpha,
            linewidths=0
        )

        # white gap
        ax.scatter(
            x, y,
            s=gap_size,
            c='white',
            alpha=alpha,
            linewidths=0
        )

        # colored center
        ax.scatter(
            x, y,
            s=size,
            c=[colors[i]],
            alpha=alpha,
            linewidths=0
        )

    """for trajectory in np.transpose(generated, axes=(1,0,2))[::30]:
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='Black')"""

    # --------------------
    # Limits
    # --------------------
    min_x = min(data_points[:, :, 0].min(), generated_points[:, :, 0].min())
    max_x = max(data_points[:, :, 0].max(), generated_points[:, :, 0].max())
    min_y = min(data_points[:, :, 1].min(), generated_points[:, :, 1].min())
    max_y = max(data_points[:, :, 1].max(), generated_points[:, :, 1].max())
    span_x = max_x - min_x
    span_y = max_y - min_y
    ax.set_xlim(min_x - span_x*0.1, max_x + span_x*0.1)
    ax.set_ylim(min_y - span_y*0.1, max_y + span_y*0.1)

    # --------------------
    # Clean axes
    # --------------------
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_frame_on(False)
    ax.patch.set_alpha(0)

    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualizeresult(
    dataset,
    results: torch.Tensor,
    save_path: str,
    dim_reduction,
    highlight_times: Optional[List[float]] = None,
    visualization_batch: int = 200
    ):
    """
    Visualize 2D trajectories with mass-based coloring.

    Args:
        dataset: object with method sample_particles_batch(batch_size)
        results: torch.Tensor of shape [T, B, D+1] (last dim = mass)
        save_path: str, path to save the figure
        highlight_times: list of time points to highlight
        visualization_batch: number of particles to sample for background
    """
    # Convert results to CPU NumPy arrays once
    results_np = results.detach().cpu().numpy()
    x_traj = results_np[..., :-1]  # [T, B, D]
    m_traj = results_np[..., -1:]   # [T, B]

    num_timesteps, batch_size, dim = x_traj.shape

    if dim > 2:
        x_traj_flat = x_traj.reshape(-1, dim)

        sample_list = []
        all_data = dataset.get_all_particles_batch()
        for key in all_data:
            sample_list.append(all_data[key])
        sample_all = torch.cat(sample_list, dim=0)
        sample_np = sample_all.cpu().detach().numpy()

        if dim_reduction == "umap":
            umap_reducer = umap.UMAP(n_components=2)
            umap_reducer.fit(sample_np)
            x_traj_flat_2d = umap_reducer.transform(x_traj_flat)
            reducer_obj = umap_reducer
        elif dim_reduction == "pca":
            pca_reducer = PCA(n_components=2)
            pca_reducer.fit(sample_np)
            x_traj_flat_2d = pca_reducer.fit_transform(x_traj_flat)
            reducer_obj = pca_reducer
        elif dim_reduction == "raw":
            class Reducer:
                def transform(self, x):
                    return x[..., :2]
            reducer_obj = Reducer()
            x_traj_flat_2d = reducer_obj.transform(x_traj_flat)
        else:
            raise ValueError

        x_traj_2d = x_traj_flat_2d.reshape(num_timesteps, batch_size, 2)
    else:
        x_traj_2d = x_traj
        reducer_obj = None

    results_2d = np.concatenate([x_traj_2d[:, :visualization_batch], m_traj[:, :visualization_batch]], axis=-1)

    # Sample background particles from dataset
    dictionary = dataset.sample_particles_batch(visualization_batch)
    sampled_data = list(dictionary.values())
    data_points = np.zeros([len(sampled_data), visualization_batch, 2])
    for i, particle in enumerate(sampled_data):
        particle_np = particle.detach().cpu().numpy()
        if dim > 2 and reducer_obj is not None:
            particle_2d = reducer_obj.transform(particle_np)
        else:
            particle_2d = particle_np
        data_points[i] = particle_2d

    plot_combined_data(data_points, results_2d, output_file=save_path, highlight_times=highlight_times)