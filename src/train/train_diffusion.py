import torch
import src.utils.utils as utils
import os

def train_diffusion(diffusion, vae, optimizer, train_loader, device, run, num_epochs, window_size, latent_channels):
    for epoch in range(num_epochs):
        for i, (x0, cond) in enumerate(train_loader):
            x0 = x0.to(device)
            cond = cond.to(device)

            # Get latent representations from your VAE.
            latent_seq = utils.encode_sequence_with_vae(vae, x0)
            latent_cond = utils.encode_sequence_with_vae(vae, cond)

            loss = diffusion.train_step(optimizer, latent_seq, latent_cond)
            if run:
                run.log({"loss": loss})

            # Free memory after each step
            del x0, cond, latent_seq, latent_cond
            torch.cuda.empty_cache()
        print(f"Epoch [{epoch+1}/{num_epochs}] : Last loss = {loss:.4f}")

        if not os.path.exists("/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints"):
            os.makedirs("/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints")

        torch.save(diffusion.model.state_dict(), "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints/diffusion_model.pt")
        torch.save(optimizer.state_dict(), "/home/ensta/ensta-lachevre/climate-uncertainty-diffusion/checkpoints/diffusion_optimizer.pt")

        # Generate sample visualization only at end of epoch
        data_iter = iter(train_loader)  # Reset iterator
        utils.sample_and_visualize(epoch, diffusion, vae, data_iter, window_size, latent_channels, device, run)



