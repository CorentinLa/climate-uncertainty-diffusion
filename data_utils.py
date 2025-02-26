import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
from torch import tensor, float32

def open_grib_file(file_path, engine='cfgrib'):
    return xr.open_dataset(file_path, engine=engine)

def load_image(dataset, time_index, sim_id=0, variables=['t2m', 'sp', 'tcc']):
    """
    Loads an image from the dataset at the specified time index and simulation ID.

    Args:
        dataset (xarray.Dataset): Dataset containing the images.
        time_index (int): Index of the time dimension to load.
        sim_id (int, optional): Simulation ID to load. Defaults to 0.
        variables (list, optional): List of variables to include in the image. Defaults to ['t2m', 'sp', 'tcc'].
    """
    data = dataset.isel(time=time_index)
    image_r = np.array(data[variables[0]].values[sim_id])
    image_g = np.array(data[variables[1]].values[sim_id])
    image_b = np.array(data[variables[2]].values[sim_id])
    image = np.stack([image_r/np.max(image_r),
                     image_g/np.max(image_g),
                     image_b/np.max(image_b)], axis=-1)
    
    return image

def load_image_sequence(dataset, start_time_index, end_time_index, sim_id=0, variables=['t2m', 'sp', 'tcc']):
    """
    Loads a sequence of images from the dataset.

    Args:
        dataset (xarray.Dataset): Dataset containing the images.
        start_time_index (int): Index of the starting time dimension to load.
        end_time_index (int): Index of the ending time dimension to load.
        sim_id (int, optional): Simulation ID to load. Defaults to 0.
        variables (list, optional): List of variables to include in the images. Defaults to ['t2m', 'sp', 'tcc'].

    Output : frames: (T, 3, H, W)
    """

    image_sequence = []
    for t in range(start_time_index, end_time_index):
        image = load_image(dataset, t, sim_id, variables)
        image_sequence.append(image)
    return np.stack(image_sequence, axis=0)

    

def export_images(dataset, output_folder, variables=['t2m', 'sp', 'tcc']):
    """
    Exports all images from the dataset to the specified output folder.

    Args:
        dataset (xarray.Dataset): Dataset containing the images.
        output_folder (str): Folder where the images will be saved.
        variables (list, optional): List of variables to include in the images. Defaults to ['t2m', 'sp', 'tcc'].
    """

    for i in range(len(dataset.time)):
        for sim in range(dataset.dims['number']):
            image = load_image(dataset, i, sim, variables)
            save_image(image, f"{output_folder}/{sim}/sim_{sim}_time_{i}.png", verbose=False)

def save_image(image, output_path, is_jpg=False, quality=95, verbose=True):
    """
    Saves an image to the specified path, creating directories if they do not exist.

    Args:
        image (numpy.ndarray): Image data to save.
        output_path (str): Full path where the image will be saved.
        is_jpg (bool, optional): Whether to save as JPEG. If False, saves as PNG. Defaults to False.
        quality (int, optional): JPEG quality (1-95). Defaults to 95.
    """
    # Extract the directory from the output path
    directory = os.path.dirname(output_path)
    
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}") if verbose else None
    
    # Save the image
    if is_jpg:
        plt.imsave(output_path, image, format='jpg', quality=quality)
        print(f"Saved JPEG image to: {output_path}") if verbose else None
    else:
        plt.imsave(output_path, image, format='png')
        print(f"Saved PNG image to: {output_path}") if verbose else None

def plot_variables(dataset, time_index, output_path=None, cmap='coolwarm', sim_id=0, variables=['t2m', 'sp', 'tcc']):
    """
    Plots the variables at the specified time index and simulation ID.

    Args:
        dataset (xarray.Dataset): Dataset containing the images.
        time_index (int): Index of the time dimension to plot.
        sim_id (int, optional): Simulation ID to plot. Defaults to 0.
        variables (list, optional): List of variables to include in the plot. Defaults to ['t2m', 'sp', 'tcc'].
    """
    data = dataset.isel(time=time_index)
    fig, axs = plt.subplots(1, len(variables), figsize=(15, 5))
    for i, var in enumerate(variables):
        axs[i].imshow(data[var].values[sim_id], cmap=cmap)
        axs[i].set_title(var)
        axs[i].axis('off')
    plt.show()
    if output_path:
        plt.savefig(output_path)

def plot_variables_from_image(image, output_path=None, cmap='coolwarm', variables=['Var 1', 'Var 2', 'Var 3']):
    """
    Plots the variables from an image.

    Args:
        image (numpy.ndarray): Image data to plot.
        output_path (str): Path where the plot will be saved.
        variables (list, optional): List of variables to include in the plot. Defaults to ['Var 1', 'Var 2', 'Var 3'].
    """
    fig, axs = plt.subplots(1, len(variables), figsize=(15, 5))
    for i, var in enumerate(variables):
        axs[i].imshow(image[:,:, i], cmap=cmap)
        axs[i].set_title(var)
        axs[i].axis('off')
    plt.show()
    if output_path:
        plt.savefig(output_path)

class ImageSequenceDataset(Dataset):
    def __init__(self, dataset, window_size, overlap=0, variables=['t2m', 'sp', 'tcc'], skip_last=True):
        """
        dataset (xarray.Dataset)
        window_size (int): Number of timestep per window.
        overlap (int, optional): Overlapping between windows. Defaults à 0.
        variables (list, optional): Liste des variables à inclure dans les images. Defaults à ['t2m', 'sp', 'tcc'].
        skip_last (bool, optional): Whether to skip the last simulation to keep for validation. Defaults to True.
        """


        self.dataset = dataset
        self.window_size = window_size
        self.overlap = overlap
        self.variables = variables
        self.total_time = dataset.dims['time']
        self.number_sim = dataset.dims['number'] - 1 - skip_last # keep one simulation for the validation
        self.step = window_size - overlap
        self.num_windows = (self.total_time - overlap) // self.step
        self.samples = []
        self.conditionner = []

        # Create a list of samples (sim_id, start_time, end_time) to have batches with elements from every simulation.
        for sim_id in range(self.number_sim):
            for w in range(self.num_windows):
                start = w * self.step
                end = min(start + window_size, self.total_time)
                self.samples.append((sim_id, start, end))
        
        for w in range(self.num_windows):
            start = w * self.step
            end = min(start + window_size, self.total_time)
            self.conditionner.append((start, end))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        sim_id, start, end = self.samples[idx]
        frames = load_image_sequence(self.dataset, start, end, sim_id=sim_id, variables=self.variables)

        if frames.shape[0] < self.window_size:
            padding = self.window_size - frames.shape[0]
            pad_frames = np.tile(frames[-1:], (padding, 1, 1, 1))
            frames = np.concatenate([frames, pad_frames], axis=0)

        frames = tensor(frames, dtype=float32).permute(0, 3, 1, 2) # (T, 3, H, W)

        start, end = self.conditionner[idx % len(self.conditionner)]

        frames_cond = load_image_sequence(self.dataset, start, end, sim_id=self.number_sim, variables=self.variables)

        if frames_cond.shape[0] < self.window_size:
            padding = self.window_size - frames_cond.shape[0]
            pad_frames = np.tile(frames_cond[-1:], (padding, 1, 1, 1))
            frames_cond = np.concatenate([frames_cond, pad_frames], axis=0)

        frames_cond = tensor(frames_cond, dtype=float32).permute(0, 3, 1, 2) # (T, 3, H, W)

        return frames, frames_cond