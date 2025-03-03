import torch
import wandb
from matplotlib import pyplot as plt
import os

def encode_sequence_with_vae(vae, frames):
    # frames: (batch, T, 3, H, W)
    latent_sequence = []
    for t in range(frames.shape[1]):
        frame_latent = vae.encode_image(frames[:, t])
        latent_sequence.append(frame_latent)
    # Stack across time => shape (batch, T, latent_channels, h, w)
    return torch.stack(latent_sequence, dim=1)

def decode_sequence_with_vae(vae, latent_seq):
    # latent_seq: (batch, T, latent_channels, h, w)
    decoded_frames = []
    for t in range(latent_seq.shape[1]):
        decoded_frame = vae.decode_latents(latent_seq[:, t])
        decoded_frames.append(decoded_frame)
    return torch.stack(decoded_frames, dim=1)  # (batch, T, 3, H, W)

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

@torch.no_grad() 
def sample_and_visualize(epoch, diffusion, vae, data_iter, window_size, latent_channels, device, run):
    # Free cache first
    torch.cuda.empty_cache()
    
    # Process a smaller batch for visualization (just 8 samples)
    try:
        truth_sample, cond_sample = next(data_iter)
        # Only use a subset of the batch
        truth_sample = truth_sample[:8].to(device)
        cond_sample = cond_sample[:8].to(device)
        
        latent_truth = encode_sequence_with_vae(vae, truth_sample).squeeze(0)
        latent_cond = encode_sequence_with_vae(vae, cond_sample).squeeze(0)
        
        # Sample with reduced steps for memory efficiency
        sampled_latent = diffusion.sample(latent_cond, (window_size*latent_channels, 45, 90))

        # Process only one sample for visualization
        sample_idx = 0

        truth_recon = decode_sequence_with_vae(
            vae, latent_truth[sample_idx:sample_idx+1].view(1, window_size, latent_channels, 45, 90)
        )
        
        cond_recon = decode_sequence_with_vae(
            vae, latent_cond[sample_idx:sample_idx+1].view(1, window_size, latent_channels, 45, 90) 
        )
        
        gen_recon = decode_sequence_with_vae(
            vae, 
            latent_cond[sample_idx:sample_idx+1] + 
            sampled_latent[sample_idx:sample_idx+1].view(1, window_size, latent_channels, 45, 90)
        )
        
        truth_frame = truth_recon[0].permute(0, 2, 3, 1).cpu().numpy()
        cond_frame = cond_recon[0].permute(0, 2, 3, 1).cpu().numpy()
        gen_frame = gen_recon[0].permute(0, 2, 3, 1).cpu().numpy()


        if run:
            # Log individual variable visualizations
            run.log({
                "t2m_comparison": [
                    wandb.Image(cond_frame[0, :, :, 2], caption="Conditioning", 
                            mode="L"),
                    wandb.Image(truth_frame[0, :, :, 2], caption="Ground Truth", 
                            mode="L"),
                    wandb.Image(gen_frame[0, :, :, 2], caption="Generated", 
                            mode="L")
                ]
            })
            
            # v10 (index 1)
            run.log({
                "v10_comparison": [
                    wandb.Image(cond_frame[0, :, :, 1], caption="Conditioning", 
                            mode="L"),
                    wandb.Image(truth_frame[0, :, :, 1], caption="Ground Truth", 
                            mode="L"),
                    wandb.Image(gen_frame[0, :, :, 1], caption="Generated", 
                            mode="L")
                ]
            })
            
            # u10 (index 0)
            run.log({
                "u10_comparison": [
                    wandb.Image(cond_frame[0, :, :, 0], caption="Conditioning", 
                            mode="L"),
                    wandb.Image(truth_frame[0, :, :, 0], caption="Ground Truth", 
                            mode="L"),
                    wandb.Image(gen_frame[0, :, :, 0], caption="Generated", 
                            mode="L")
                ]
            })
        
        # Save plot
        if not os.path.exists("/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/data/results/diffusion"):
            os.makedirs("/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/data/results/diffusion")

        plot_variables_from_image(
            gen_frame[0, :, :, :], 
            output_path=f"/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/data/results/diffusion/sampled_frames_{epoch}.png"
        )
        
        # Explicitly delete tensors and call garbage collector
        del truth_sample, cond_sample, latent_truth, latent_cond, sampled_latent
        del truth_recon, cond_recon, gen_recon, truth_frame, cond_frame, gen_frame
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        torch.cuda.empty_cache()