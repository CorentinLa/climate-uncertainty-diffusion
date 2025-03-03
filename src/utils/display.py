import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl

from data_utils import load_pickle
from utils import verif_path

# Enable anti-aliasing globally
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['lines.antialiased'] = True
mpl.use('Agg')

cbar = None

ABSOLUTE_ZERO = -273.15  # in degrees celsius


def kelvin_to_celsius(temp):
    return temp + ABSOLUTE_ZERO


def compute_animation_for_vectors(vectors, save_path, titles=None, cbar_label=None, lat_ext=[-90, 90], lon_ext=[-180, 180]):
    """

    :param vectors: (n_scalars, time, dimensions(lat and lon), lat, lon)
    :param lat_ext:
    :param lon_ext:
    :param save_path:
    :return:
    """
    n_subplots = len(vectors)
    n_timesteps = vectors[0].shape[0]
    colormap = 'viridis'

    H, W = vectors[0].shape[-2:]
    lat = np.linspace(lat_ext[0], lat_ext[1], H)  # H points from -90 to 90 (latitude)
    lon = np.linspace(lon_ext[0], lon_ext[1], W)  # W points from -180 to 180 (longitude)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Set up the figure and axis
    fig, axs = plt.subplots(nrows=n_subplots, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8), dpi=300)

    if n_subplots == 1:
        axs = [axs]

    # Compute the wind magnitude
    magnitudes = [np.sqrt(vectors[i][:, 0] ** 2 + vectors[i][:, 1] ** 2) for i in range(n_subplots)]

    # To have a consistent colorbar over all the busplots
    vmin = min([s_.min() for s_ in magnitudes])
    vmax = min([s_.max() for s_ in magnitudes])
    normalizer = plt.Normalize(vmin, vmax)
    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs)
    cbar.set_label(cbar_label)

    def update_vectors(frame):

        for i, ax in enumerate(axs):
            ax.clear()
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
            ax.coastlines()
            ax.set_title(titles[i])  # Customize title as needed

            # Plot the wind vectors with color based on magnitude
            ax.quiver(
                lon2d, lat2d,
                vectors[i][frame, 0], vectors[i][frame, 1],
                magnitudes[i][frame],
                transform=ccrs.PlateCarree(),
                # scale=1,
                scale_units='xy',
                width=0.0018,
                cmap=colormap
            )

    # Create an animation
    ani = FuncAnimation(fig, update_vectors, frames=n_timesteps, interval=200, blit=False)

    # Save the animation with high quality
    ani.save(save_path, writer="ffmpeg", fps=1, dpi=300, bitrate=5000)
    cbar = None
    plt.close()


def compute_animation_for_scalars(scalars, save_path, titles=None, cbar_label=None, lat_ext=[-90, 90], lon_ext=[-180, 180]):
    """

    :param scalars: (n_scalars, time, lat, lon)
    :param lat_ext:
    :param lon_ext:
    :param save_path:
    :return:
    """
    n_scalars = len(scalars)
    n_timesteps = scalars[0].shape[0]
    colormap = 'coolwarm'

    extent = lon_ext + lat_ext

    # Set up the figure and axis
    fig, axs = plt.subplots(nrows=n_scalars, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8), dpi=300)

    if n_scalars == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='none')
        ax.coastlines()
        ax.set_title(titles[i])  # Customize title as needed

    # To have a consistent colorbar over all the busplots
    vmin = min([s_.min() for s_ in scalars])
    vmax = min([s_.max() for s_ in scalars])
    normalizer = plt.Normalize(vmin, vmax)
    cbar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=colormap)
    cbar = fig.colorbar(cbar_mappable, ax=axs)
    cbar.set_label(cbar_label)

    def update_scalars(frame):
        # ax.clear()
        for i, ax in enumerate(axs):
            ax.imshow(scalars[i][frame], origin='lower', extent=extent, cmap=colormap, norm=normalizer)

    # Create an animation
    ani = FuncAnimation(fig, update_scalars, frames=n_timesteps, interval=200, blit=False)

    # Save the animation with high quality
    ani.save(save_path, writer="ffmpeg", fps=1, dpi=300, bitrate=5000)
    plt.close()


# Load the NetCDF file
file_path = "evaluation/fitted_velocity_2016-01-01_to_2016-01-14.pkl"  # Replace with your NetCDF file path
data = load_pickle(file_path)

extra = f'_epoch95'

levels = ["z", "t", "t2m", "u10", "v10"]
levels_idx = {level_: i for i, level_ in enumerate(levels)}

DO_TEMP = False
DO_WIND = True
DO_REST = False

for ts in data:
    #  Temperature
    if DO_TEMP:
        for level in ['t', 't2m']:
            idx = levels_idx[level]
            gt = kelvin_to_celsius(data[ts]['u_gt'][:, idx])
            pred = kelvin_to_celsius(data[ts]['u_pred_w_bias'][:, idx])
            verif_path(f'evaluation/animations/{level}')
            compute_animation_for_scalars(
                [gt, pred], f'evaluation/animations/{level}/{ts}{extra}.mp4', titles=[f'{level}_gt', f'{level}_pred'], cbar_label='Temperature °C'
                )

    #  Wind
    if DO_WIND:
        #  TODO : see https://confluence.ecmwf.int/display/CKB/Copernicus+Arctic+Regional+Reanalysis+%28CARRA%29%3A+Data+User+Guide#CopernicusArcticRegionalReanalysis(CARRA):DataUserGuide-Variablesat10-metreheight
        wind_gt = np.stack((data[ts]['u_gt'][:, levels_idx['v10']], data[ts]['u_gt'][:, levels_idx['u10']]), axis=1)
        wind_pred = np.stack((data[ts]['u_pred_w_bias'][:, levels_idx['v10']], data[ts]['u_pred_w_bias'][:, levels_idx['u10']]), axis=1)

        verif_path(f'evaluation/animations/wind')
        compute_animation_for_vectors(
            [wind_gt, wind_pred], f'evaluation/animations/wind/{ts}{extra}.mp4', titles=[f'wind_gt', f'wind_pred'], cbar_label='Wind speed m/s'
            )

    #  Rest
    if DO_REST:
        for level in ['z']:
            idx = levels_idx[level]
            gt = data[ts]['u_gt'][:, idx]
            # compute_animation_for_scalar(t2m_gt, lat, lon, f'evaluation/animations/t2m_gt_animation_{ts}.mp4')
            pred = data[ts]['u_pred_w_bias'][:, idx]
            # compute_animation_for_scalar(t2m_pred, lat, lon, f'evaluation/animations/t2m_pred_w_bias_animation_{ts}.mp4')
            verif_path(f'evaluation/animations/{level}')
            compute_animation_for_scalars([gt, pred], f'evaluation/animations/{level}/{ts}{extra}.mp4', titles=[f'{level}_gt', f'{level}_pred'])
