import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
from torch import tensor, float32
from tqdm import tqdm
import torch
import src.utils.utils as utils
import json

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

def compute_residual_variance(data_loader, vae):
    """
    Compute the residual variance between the latent space of the input and the latent space of the conditionning.
    """
    # Process a decent number of batches
    residual_samples = []
    with torch.no_grad():
        for i, (x0, cond) in tqdm(enumerate(data_loader)):
            x0 = x0.to('cuda')
            cond = cond.to('cuda')
            latent_x0 = utils.encode_sequence_with_vae(vae, x0)
            latent_cond = utils.encode_sequence_with_vae(vae, cond)
            residual = latent_x0 - latent_cond
            residual_samples.append(residual.cpu())
            if i >= 30:  # Process 10 batches
                break
                
    residuals = torch.cat(residual_samples, dim=0)
    # Compute overall std across all dimensions
    std = residuals.std().item()
    print(f"Computed residual std: {std}")
    return std

def compute_statistics(dataset, compute_stats=True):
    """
    Compute the mean, max and min of each variable in the dataset for each time step.
    Returns a structured dictionary with statistics organized by time and variable.
    """

    if not compute_stats:
        path = '/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/data/stats.json'
        with open(path, 'r') as f:
            stats = json.load(f)
        return stats


    stats_global = {}

    for sim_id in tqdm(range(dataset.dims['number']-2)):
        stats = {}

        # Convert numpy datetime64 objects to strings for JSON serialization
        date_strings = [str(date) for date in dataset['time'].values]
        
        # Initialize the stats dictionary with dates
        for i, date in enumerate(date_strings):
            stats[date] = {}
            
        # Compute stats for each variable at each time
        for i in tqdm(range(dataset.dims['time']), desc="Computing statistics"):
            current_date = date_strings[i]
            
            for var in dataset.data_vars:
                data = dataset[var].values[sim_id, i]
                stats[current_date][var] = float(np.mean(data))

        stats_global[sim_id] = stats


    # Export to JSON
    output_path = '/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/data/stats.json'
    with open(output_path, 'w+') as f:
        json.dump(stats_global, f, indent=2)
    
    print(f"Statistics saved to {output_path}")
    return stats_global




class ImageSequenceDataset(Dataset):
    def __init__(self, dataset, window_size, overlap=0, variables=['t2m', 'sp', 'tcc'], skip_last=True):
        """
        dataset (xarray.Dataset)
        window_size (int): Number of timestep per window.
        overlap (int, optional): Overlapping between windows. Defaults à 0.
        variables (list, optional): Liste des variables à inclure dans les images. Defaults à ['t2m', 'sp', 'tcc'].
        skip_last (bool, optional): Whether to skip the last simulation to keep for validation. Defaults to True.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.variables = variables
        self.total_time = dataset.dims['time']
        self.number_sim = dataset.dims['number'] - 1 - skip_last
        self.step = window_size - overlap
        self.num_windows = (self.total_time - overlap) // self.step
        self.samples = []
        self.conditionner = []

        # Create sample indices
        for sim_id in range(self.number_sim):
            for w in range(self.num_windows):
                start = w * self.step
                end = min(start + window_size, self.total_time)
                self.samples.append((sim_id, start, end))
        
        for w in range(self.num_windows):
            start = w * self.step
            end = min(start + window_size, self.total_time)
            self.conditionner.append((start, end))


        # Preload all data into memory
        print("Preloading all data into memory - this may take a moment...")
        
        # Extract all variables at once for all simulations
        all_data = {}
        for var in tqdm(variables, desc="Loading variables"):
            # Load all time steps and simulations for this variable
            all_data[var] = dataset[var].values
        
        # Precompute normalization factors
        self.max_values = {var: np.max(all_data[var]) for var in variables}
        
        # Preprocess all frames for all simulations
        self.preprocessed_frames = {}
        for sim_id in tqdm(range(self.number_sim + 1 + 1), desc="Preprocessing frames"):  # Include validation sim
            self.preprocessed_frames[sim_id] = []
            for t in range(self.total_time):
                # Create normalized image for this frame
                image = np.stack([
                    all_data[variables[0]][sim_id, t] / self.max_values[variables[0]],
                    all_data[variables[1]][sim_id, t] / self.max_values[variables[1]],
                    all_data[variables[2]][sim_id, t] / self.max_values[variables[2]]
                ], axis=-1)
                self.preprocessed_frames[sim_id].append(image)
        
        print("Data preloading complete!")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sim_id, start, end = self.samples[idx]
        
        # Get precomputed frames
        frames = self.preprocessed_frames[sim_id][start:end]
        
        # Handle padding if needed
        if len(frames) < self.window_size:
            padding = self.window_size - len(frames)
            pad_frames = [frames[-1]] * padding
            frames = frames + pad_frames
            
        # Stack and convert to tensor
        frames = np.stack(frames, axis=0)
        frames = tensor(frames, dtype=float32).permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        # Get conditioning frames
        start, end = self.conditionner[idx % len(self.conditionner)]
        cond_sim_id = self.number_sim  # Use validation sim
        
        frames_cond = self.preprocessed_frames[cond_sim_id][start:end]
        
        # Handle padding if needed
        if len(frames_cond) < self.window_size:
            padding = self.window_size - len(frames_cond)
            pad_frames = [frames_cond[-1]] * padding
            frames_cond = frames_cond + pad_frames
            
        # Stack and convert to tensor
        frames_cond = np.stack(frames_cond, axis=0)
        frames_cond = tensor(frames_cond, dtype=float32).permute(0, 3, 1, 2)  # (T, 3, H, W)

        return frames, frames_cond


class TestImageSequenceDataset(Dataset):
    def __init__(self, dataset, window_size, overlap=0, variables=['t2m', 'sp', 'tcc']):
        """
        Dataset that provides only the last simulation for testing purposes.
        
        Args:
            dataset (xarray.Dataset): Dataset containing the climate data
            window_size (int): Number of timesteps per window
            overlap (int, optional): Overlapping between windows. Defaults to 0.
            variables (list, optional): List of variables to include. Defaults to ['t2m', 'sp', 'tcc'].
        """
        self.window_size = window_size
        self.overlap = overlap
        self.variables = variables
        self.total_time = dataset.dims['time']
        # We only use the last simulation
        self.sim_id = dataset.dims['number'] - 2
        self.step = window_size - overlap
        self.num_windows = (self.total_time - overlap) // self.step
        self.time_indices = []
        
        # Create time indices for windows
        for w in range(self.num_windows):
            start = w * self.step
            end = min(start + window_size, self.total_time)
            self.time_indices.append(start)
        
        print("Preloading test data into memory...")
        
        # Extract all variables for the last simulation
        all_data = {}
        for var in tqdm(variables, desc="Loading variables"):
            all_data[var] = dataset[var].values[self.sim_id]
        
        # Precompute normalization factors
        self.max_values = {var: np.max(all_data[var]) for var in variables}
        # Preprocess all frames for the test simulation
        self.preprocessed_frames = []
        for t in tqdm(range(self.total_time), desc="Preprocessing frames"):
            # Create normalized image for this frame
            image = np.stack([
                all_data[variables[0]][t] / self.max_values[variables[0]],
                all_data[variables[1]][t] / self.max_values[variables[1]],
                all_data[variables[2]][t] / self.max_values[variables[2]]
            ], axis=-1)
            self.preprocessed_frames.append(image)
        
        print("Test data preloading complete!")

    def __len__(self):
        return len(self.time_indices)
    
    def __getitem__(self, idx):
        start_time = self.time_indices[idx]
        end_time = min(start_time + self.window_size, self.total_time)
        
        # Get precomputed frames
        frames = self.preprocessed_frames[start_time:end_time]
        
        # Handle padding if needed
        if len(frames) < self.window_size:
            padding = self.window_size - len(frames)
            pad_frames = [frames[-1]] * padding
            frames = frames + pad_frames
            
        # Stack and convert to tensor
        frames = np.stack(frames, axis=0)
        frames = tensor(frames, dtype=float32).permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        return frames, start_time
