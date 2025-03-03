from tqdm import tqdm
import torch
import src.utils.utils as utils
import os
import json

def test_diffusion(diffusion, vae, data_loader, device, window_size, latent_channels, variables, max_values, n_simulations=1, time_values=None):
    """Test the diffusion model on the given data loader.
    
    Args:
        diffusion: The diffusion model
        vae: The variational autoencoder
        data_loader: DataLoader for test data
        device: Device to run on
        window_size: Size of the time window
        latent_channels: Number of latent channels
        variables: List of variable names
        max_values: Dictionary of maximum values for each variable
        n_simulations: Number of simulations to run
        time_values: List of time values (datetime strings) corresponding to indices
    """
    # Ensure we have time values for proper alignment
    if time_values is None:
        print("Warning: No time_values provided. Using integer indices as keys.")
        use_time_strings = False
    else:
        use_time_strings = True

    for k in tqdm(range(n_simulations), desc="Testing the diffusion model..."):
        stats_diffusion = {}

        for i, (cond, time) in enumerate(data_loader):
            cond = cond.to(device)

            latent_cond = utils.encode_sequence_with_vae(vae, cond)

            # Sample from the diffusion model
            sampled_latent = diffusion.sample(latent_cond, (window_size*latent_channels, 45, 90))

            # Decode the sampled latent
            gen_recon = utils.decode_sequence_with_vae(vae, latent_cond+sampled_latent.view(cond.shape[0], window_size, latent_channels, 45, 90)) # (batch, T, 3, H, W)

            # Free memory after each step
            del cond, latent_cond, sampled_latent
            torch.cuda.empty_cache()

            for b in range(gen_recon.shape[0]):
                for t_idx in range(time[b], time[b] + window_size):
                    # Use datetime string as key if available, otherwise use integer index
                    t_key = time_values[t_idx] if use_time_strings else t_idx
                    
                    if t_key not in stats_diffusion:
                        stats_diffusion[t_key] = {}
                        for var in variables:
                            stats_diffusion[t_key][var] = []
                
                    for i, var in enumerate(variables):
                        var_data = gen_recon[b, t_idx - time[b], i].cpu().numpy()
                        stats_diffusion[t_key][var].append(float(var_data.mean()))
        
        # Calculate averages
        for t_key in stats_diffusion:
            for var in variables:
                stats_diffusion[t_key][var] = float(sum(stats_diffusion[t_key][var]) / len(stats_diffusion[t_key][var]) * max_values[var])

        # Dump stats to json
        output_path = "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/data/stats_diffusion.json"

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                all_stats = json.load(f)
                all_stats = {int(k): v for k, v in all_stats.items()}
                next_id = max(all_stats.keys()) + 1 if all_stats else 0
        else:
            all_stats = {}
            next_id = 0

        # Add the new statistics with the unique ID
        all_stats[next_id] = stats_diffusion

        # Write the updated data back to the JSON file
        with open(output_path, 'w+') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"Statistics saved to {output_path} with ID {next_id}")